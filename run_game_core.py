# run_game_core.py
import argparse, os, glob, json, math
import cv2, numpy as np, torch
from depth_anything_v2.dpt import DepthAnythingV2
import shutil, subprocess


# ---------- 幫手函式 ----------
def pixel_to_cam3d(u, v, z_m, fx, fy, cx, cy):
    # 相機座標系：Z 朝前，X 右，Y 下（之後在 Unity 可再做軸向對齊）
    X = (u - cx) / fx * z_m
    Y = (v - cy) / fy * z_m
    return [float(X), float(Y), float(z_m)]

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    dmin, dmax = float(np.min(depth)), float(np.max(depth))
    if dmax - dmin < 1e-6:
        return np.zeros_like(depth, dtype=np.uint8)
    d = (depth - dmin) / (dmax - dmin)
    return (d * 255.0).astype(np.uint8)

def colorize_gray(img8: np.ndarray) -> np.ndarray:
    # BGR 彩色深度（用 OpenCV colormap，快）
    return cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)

def dp_simplify(cnt: np.ndarray, eps_ratio: float) -> np.ndarray:
    # cnt: (N,1,2) 或 (N,2)
    c = cnt.reshape(-1, 2)
    peri = cv2.arcLength(c, False)
    eps = max(1.0, eps_ratio * peri)
    approx = cv2.approxPolyDP(c, eps, False).reshape(-1, 2)
    return approx

def polyline_length(px: np.ndarray) -> float:
    if len(px) < 2: return 0.0
    return float(np.linalg.norm(np.diff(px, axis=0), axis=1).sum())

def split_by_turn(poly: np.ndarray, max_turn_deg: float) -> list:
    """若轉折角過尖，切成多段（避免不可滑的尖角）"""
    if len(poly) < 3: return [poly]
    segs, cur = [poly[0]], []
    for i in range(1, len(poly) - 1):
        a, b, c = poly[i-1], poly[i], poly[i+1]
        v1 = a - b; v2 = c - b
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = math.degrees(math.acos(cosang))  # 0~180，越小越尖
        segs.append(b)
        if angle < (180 - max_turn_deg):  # 例如 max_turn_deg=140 → 小於40度就切
            segs.append(None)
    segs.append(poly[-1])

    out, buf = [], []
    for p in segs:
        if p is None:
            if len(buf) >= 2: out.append(np.array(buf, dtype=np.int32))
            buf = []
        else:
            buf.append(p.tolist())
    if len(buf) >= 2: out.append(np.array(buf, dtype=np.int32))
    return out

def sample_with_depth(poly: np.ndarray, depth8: np.ndarray, step_px=3) -> list:
    """下採樣 polyline 並附加像素深度（之後 Unity 可做高度/坡度效果）"""
    if len(poly) == 0: return []
    sampled = [poly[0]]
    acc = 0.0
    for i in range(1, len(poly)):
        d = np.linalg.norm(poly[i] - poly[i-1])
        acc += d
        if acc >= step_px:
            sampled.append(poly[i])
            acc = 0.0
    if (sampled[-1] != poly[-1]).any():
        sampled.append(poly[-1])
    pts = []
    H, W = depth8.shape[:2]
    for x, y in sampled:
        xi = int(np.clip(x, 0, W-1)); yi = int(np.clip(y, 0, H-1))
        pts.append([float(x), float(y), float(depth8[yi, xi]) / 255.0])  # z=規範化深度
    return pts

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser("Skate AR — depth->contour->polyline core")
    parser.add_argument('--video-path', type=str, help="影片路徑、資料夾或 .txt 列表；或用 'cam' 開攝像頭")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./game_outputs')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', action='store_true')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--fov-deg', type=float, default=60.0, help='假設的相機水平視角')
    parser.add_argument('--z-scale', type=float, default=5.0, help='把規範化深度[0,1]拉到可視的公尺尺度')

    # 下面是遊戲用參數
    parser.add_argument('--grad-thresh', type=int, default=40, help='深度梯度二值化門檻')
    parser.add_argument('--min-length', type=float, default=120.0, help='polyline 最短像素長度')
    parser.add_argument('--dp-eps', type=float, default=0.015, help='Douglas–Peucker 比例')
    parser.add_argument('--max-turn-deg', type=float, default=140.0, help='允許最大轉折角（越大越平滑）')
    parser.add_argument('--topk', type=int, default=16, help='每幀輸出前 k 條最長 polyline')
    parser.add_argument('--draw-edges', action='store_true', help='除輪廓外額外顯示邊緣圖')
    
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_cfgs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,192,384,768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256,512,1024,1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536,1536]},
    }

    model = DepthAnythingV2(**model_cfgs[args.encoder])
    ckpt_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(DEVICE).eval()

    # 準備輸入清單
    if args.video_path in ('cam', 'camera', 'webcam'):
        inputs = [('__camera__', 0)]
    else:
        if os.path.isfile(args.video_path):
            if args.video_path.endswith('.txt'):
                with open(args.video_path, 'r', encoding='utf-8') as f:
                    paths = [ln.strip() for ln in f if ln.strip()]
            else:
                paths = [args.video_path]
        else:
            paths = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
        inputs = [(p, p) for p in paths if os.path.splitext(p)[1].lower() in ('.mp4','.mov','.avi','.mkv','.m4v')]

    os.makedirs(args.outdir, exist_ok=True)

    for idx, (key, src) in enumerate(inputs):
        print(f'[{idx+1}/{len(inputs)}] {key}')
        cap = cv2.VideoCapture(src) if key != '__camera__' else cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f'  !! 無法開啟：{key}')
            continue

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = max(1, int(cap.get(cv2.CAP_PROP_FPS))) if key != '__camera__' else 30
        cx, cy = W * 0.5, H * 0.5
        fx = (W / (2.0 * math.tan(math.radians(args.fov_deg) * 0.5)))
        fy = fx  # 先假設像素方形

        
        if FPS is None or FPS != FPS:   # NaN
            FPS = 30
        FPS = float(FPS)
        if FPS <= 0 or FPS > 240:       # 有些影片回報 0 或怪值
            FPS = 30.0

        # 視覺化輸出
        if args.pred_only:
            out_w = W
        else:
            if args.pred_only:
                out_w = W
            else:
                panes = 3 if args.draw_edges else 2  # 左：原圖+polyline；中：可選邊緣；右：深度彩圖
                sep = 16                             # 兩個面板之間的白色分隔寬度
                out_w = W * panes + sep * (panes - 1)
        out_h = H

        base = 'camera' if key == '__camera__' else os.path.splitext(os.path.basename(key))[0]
        def make_writer(base_path_no_ext, fps, size):
            # 依序嘗試：MJPG(avi) -> XVID(avi) -> mp4v(mp4) -> avc1(mp4)
            # （先保證寫得出、播放得了；H.264 留給 ffmpeg 轉檔）
            trials = [
                ('MJPG', '.avi'),   # 幾乎保證可播
                ('XVID', '.avi'),
                ('mp4v', '.mp4'),   # 有些 Windows 不吃
                ('avc1', '.mp4'),   # OpenCV 多半沒有這個編碼器
            ]
            for fourcc_name, ext in trials:
                p = base_path_no_ext + ext
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                vw = cv2.VideoWriter(p, fourcc, float(fps), size)
                if vw.isOpened():
                    print(f"[writer] using {fourcc_name} -> {p}")
                    return vw, p
                vw.release()
            raise RuntimeError("No available codec. Install ffmpeg (libx264) or keep AVI.")

        base_no_ext = os.path.join(args.outdir, f'{base}_overlay')
        writer, video_out = make_writer(base_no_ext, FPS, (out_w, out_h))
        assert writer.isOpened(), "VideoWriter open failed — 請先用 MJPG/AVI，或安裝 ffmpeg+libx264"


        # 替換原本的 writer 建立方式
        
        assert writer.isOpened(), "VideoWriter open failed — try installing ffmpeg/libx264"

        # JSON 每幀輸出資料夾
        json_dir = os.path.join(args.outdir, f'{base}_polylines')
        os.makedirs(json_dir, exist_ok=True)

        frame_id = 0
        frames_written = 0

        # 先探一幀，避免整支影片 0 幀
        ok_probe, probe = cap.read()
        if not ok_probe or probe is None:
            print("  !! 無法從來源讀到任何畫面。請安裝 ffmpeg 或換另一支影片： sudo apt-get install -y ffmpeg")
            writer.release(); cap.release()
            print(f'  !! 已跳過：{key}')
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到開頭

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_id += 1

            with torch.no_grad():
                depth = model.infer_image(frame, args.input_size)  # float32, HxW
            depth8 = normalize_depth(depth)

            # 幾何邊界（深度不連續）
            gx = cv2.Sobel(depth8, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(depth8, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            edges8 = cv2.convertScaleAbs(mag)
            edges8 = cv2.GaussianBlur(edges8, (5,5), 0)
            _, mask = cv2.threshold(edges8, args.grad_thresh, 255, cv2.THRESH_BINARY)

            # 拓樸清理（閉合小洞 + 連接）
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # 找輪廓
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # 簡化 + 曲率切段 + 長度過濾
            polys = []
            for c in cnts:
                if len(c) < 8: continue
                simp = dp_simplify(c, args.dp_eps)
                for seg in split_by_turn(simp, args.max_turn_deg):
                    if polyline_length(seg) >= args.min_length:
                        polys.append(seg)

            # 取最長前 K
            polys.sort(key=lambda p: polyline_length(p), reverse=True)
            polys = polys[:max(1, args.topk)]

            # 視覺化：左=原圖+polyline，右=深度彩圖（可選中間再顯示邊緣）
            vis_left = frame.copy()
            for p in polys:
                cv2.polylines(vis_left, [p.astype(np.int32)], isClosed=False, color=(0,255,0), thickness=2)

            depth_color = colorize_gray(depth8)
            canvas = [vis_left, np.full((H,16,3), 255, np.uint8), depth_color]
            if args.draw_edges:
                canvas = [vis_left, np.full((H,16,3), 255, np.uint8), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                          np.full((H,16,3), 255, np.uint8), depth_color]
            out_frame = cv2.hconcat(canvas) if not args.pred_only else vis_left
            
            h, w = out_frame.shape[:2]
            if (w, h) != (out_w, out_h):
                # 非預期大小時，保守做 resize 確保正確輸出
                out_frame = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(out_frame)
            frames_written += 1

            # 匯出 JSON（每幀一檔）
            lines = [sample_with_depth(p.astype(np.int32), depth8, step_px=3) for p in polys]
            lines = [ln for ln in lines if len(ln) >= 2]
            lines_cam = []
            for ln in lines:  # ln: [ [x,y,z_norm], ... ]
                pts3 = []
                for x, y, z_norm in ln:
                    z_m = float(z_norm) * float(args.z_scale)  # 0~1 → 0~z_scale(公尺)
                    pts3.append(pixel_to_cam3d(x, y, z_m, fx, fy, cx, cy))
                lines_cam.append(pts3)

            payload = {
                "frame": frame_id,
                "width": W, "height": H, "fps": FPS,
                "polylines": [
                    {
                        "points": ln,
                        "points_cam": lines_cam[i],   # 新增：相機座標 3D
                        "length_px": float(polyline_length(np.array([[pt[0],pt[1]] for pt in ln], dtype=np.float32)))
                    } for i, ln in enumerate(lines)
                ],
                "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "fov_deg": args.fov_deg, "z_scale": args.z_scale}
            }

            with open(os.path.join(json_dir, f'{frame_id:06d}.json'), 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)

        writer.release()
        cap.release()
        print(f'  ✔ 影片輸出：{video_out}')
        print(f'  ✔ JSON 幀資料夾：{json_dir}')
        # 自動用 ffmpeg 轉成 H.264（若可用），確保 Windows/手機能播
        if shutil.which("ffmpeg") and video_out.lower().endswith(".avi"):
            mp4_out = base_no_ext + "_h264.mp4"
            cmd = [
                "ffmpeg", "-y", "-i", video_out,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                mp4_out
            ]
            try:
                print("[ffmpeg] converting AVI -> H.264 MP4 ...")
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"  ✔ 轉檔完成：{mp4_out}")
            except Exception as e:
                print(f"  !! ffmpeg 轉檔失敗：{e}")

if __name__ == '__main__':
    main()
