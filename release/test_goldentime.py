import os
import sys
import time
import math
import argparse
import glob
import cv2
import numpy as np

# ==============================================================================
# 1. 시스템 기본 파라미터 (IBS-PAS Cockpit Specification)
# ==============================================================================
CAMERA_FOV_DEG = 130.0          # 카메라 수평 화각 (Horizontal FOV)
CONF_THRESHOLD = 0.25           # 기본 신뢰도 임계값 (25%)
IOU_THRESHOLD = 0.45            # NMS IoU 임계값
MODEL_INPUT_SIZE = 640          # ONNX 모델 입력 해상도
FLIGHT_STATE = 'CRUISE'         # 비행 상태: 'CRUISE'(순항: Rc=192m) / 'TAKEOFF'(이착륙: Rc=220m)
ZOOM_LEVEL = 15.0               # 광학 줌 배율
DECAY_STEP_SEC = 3.3            # 위험도 3단계 감쇠 시간 간격

DEFAULT_PLAYLIST = [f"./test{i}.mp4" for i in range(9)]
MODEL_PATH = "./yolo11n.onnx"
OUTPUT_DIR = "./result"         # 결과 캡처 이미지 저장 폴더

# ==============================================================================
# 2. 파일 입력 파서 (CLI 전용 - tkinter 완전 배제)
# ==============================================================================
def get_target_video_files():
    """
    1. CLI 인자(-i / --input)로 전달된 파일 또는 디렉토리 확인
    2. 인자가 없으면 기본 샘플 시나리오(test0.mp4 ~ test8.mp4) 순차 분석
    """
    parser = argparse.ArgumentParser(description="IBS-PAS 3D Aerial Collision Avoidance Headless Analyzer")
    parser.add_argument('-i', '--input', type=str, default=None, help="비디오 파일 경로, 디렉토리 경로 또는 와일드카드 패턴 (*.mp4)")
    parser.add_argument('-m', '--model', type=str, default=MODEL_PATH, help="ONNX 모델 경로")
    parser.add_argument('-c', '--conf', type=float, default=CONF_THRESHOLD, help="신뢰도 임계값")
    parser.add_argument('-z', '--zoom', type=float, default=ZOOM_LEVEL, help="광학 줌 배율")
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_DIR, help="결과 캡처 이미지 저장 경로")
    parser.add_argument('--save-all', action='store_true', help="표적이 감지되지 않은 프레임도 전수 저장 (기본값: 표적 탐지 프레임만 저장)")
    args = parser.parse_args()

    selected_files = []

    if args.input:
        if os.path.isfile(args.input):
            selected_files = [args.input]
        elif os.path.isdir(args.input):
            valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
            selected_files = [
                os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                if f.lower().endswith(valid_exts)
            ]
        else:
            # 와일드카드 패턴 처리 (*.mp4)
            matched = sorted(glob.glob(args.input))
            if matched:
                selected_files = matched
            else:
                print(f"[경고] 지정된 입력 파일/패턴을 찾을 수 없습니다: {args.input}")

    if not selected_files:
        print("[안내] 입력 인자가 없어 기본 시나리오(test0.mp4 ~ test8.mp4)를 순차 분석합니다.")
        selected_files = [f for f in DEFAULT_PLAYLIST if os.path.exists(f)]

    return selected_files, args

# ==============================================================================
# 3. 3D 공간 좌표계 및 위험도 연산
# ==============================================================================
def calculate_3d_spatial_data(x1, y1, x2, y2, img_w, img_h, zoom=15.0, fov_deg=130.0):
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    
    # 거리 역산 모델 (640 정규화 기준)
    scale_factor = img_w / float(MODEL_INPUT_SIZE)
    norm_dim = max(bw, bh) / scale_factor
    distance_m = 500.0 * (100.0 / max(norm_dim, 1.0)) * (zoom / 15.0)
    
    # 방위각 (Bearing Angle: 좌측 -65° ~ 우측 +65°)
    cx = (x1 + x2) / 2.0
    bearing_deg = ((cx / float(img_w)) - 0.5) * fov_deg
    bearing_rad = math.radians(bearing_deg)
    
    # 앙각 (Elevation Angle)
    cy = (y1 + y2) / 2.0
    vert_fov_deg = fov_deg * (float(img_h) / float(img_w))
    elevation_deg = -((cy / float(img_h)) - 0.5) * vert_fov_deg
    elevation_rad = math.radians(elevation_deg)
    
    # 3D 직교 좌표계 (X: 좌우 수평, Y: 상하고도, Z: 전방거리)
    coord_z = distance_m * math.cos(elevation_rad) * math.cos(bearing_rad)
    coord_x = distance_m * math.cos(elevation_rad) * math.sin(bearing_rad)
    coord_y = distance_m * math.sin(elevation_rad)
    
    return {
        "distance_m": distance_m,
        "bearing_deg": bearing_deg,
        "elevation_deg": elevation_deg,
        "coord_3d": (coord_x, coord_y, coord_z),
        "center_px": (int(cx), int(cy))
    }

def evaluate_risk_level(distance_m, rc_threshold):
    """4단계 위험 등급 및 테두리 색상 분류"""
    if distance_m <= rc_threshold:
        return 4, "Level 4 (Critical)", (0, 0, 255)      # 적색
    elif distance_m <= 500.0:
        return 3, "Level 3 (Warning)", (0, 152, 255)     # 주황색
    elif distance_m <= 750.0:
        return 2, "Level 2 (Caution)", (0, 235, 255)     # 황색
    else:
        return 1, "Level 1 (Safe)", (0, 255, 0)          # 녹색

# ==============================================================================
# 4. YOLO11 ONNX 추론 엔진 (GPU E2E 및 CPU Fallback 호환)
# ==============================================================================
class YOLO11Detector:
    def __init__(self, onnx_model_path):
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def preprocess(self, frame):
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, w, h
    
    def infer(self, frame, conf_thresh=CONF_THRESHOLD):
        blob, orig_w, orig_h = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        output = outputs[0]
        
        detections = []
        dims = output.shape
        scale_x = orig_w / float(MODEL_INPUT_SIZE)
        scale_y = orig_h / float(MODEL_INPUT_SIZE)
        
        # 1. End-to-End NMS 구조 [1, N, 6] 파싱
        if len(dims) == 3 and dims[2] == 6:
            for det in output[0]:
                score = float(det[4])
                if score >= conf_thresh:
                    x1, y1, x2, y2 = det[0] * scale_x, det[1] * scale_y, det[2] * scale_x, det[3] * scale_y
                    cls_id = int(det[5])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class_id": cls_id
                    })
            return detections

        # 2. Raw Anchor 구조 [1, 84, 8400] 파싱
        if len(dims) == 3:
            raw_data = output[0]
            if dims[1] < dims[2]:
                raw_data = raw_data.T
                
            boxes = []
            confidences = []
            class_ids = []
            
            for row in raw_data:
                scores = row[4:]
                max_score = np.max(scores)
                if max_score >= conf_thresh:
                    cx, cy, w, h = row[0] * scale_x, row[1] * scale_y, row[2] * scale_x, row[3] * scale_y
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    boxes.append([int(x1), int(y1), int(w), int(h)])
                    confidences.append(float(max_score))
                    class_ids.append(int(np.argmax(scores)))
                    
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, IOU_THRESHOLD)
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes[idx]
                    detections.append({
                        "bbox": [bx, by, bx + bw, by + bh],
                        "score": confidences[idx],
                        "class_id": class_ids[idx]
                    })
        return detections

# ==============================================================================
# 5. 캡처 이미지 내 HUD 및 텔레메트리 렌더러
# ==============================================================================
class CockpitHUDVisualizer:
    def __init__(self):
        self.global_risk_level = 1
        self.risk_expire_time = 0.0

    def update_decay_system(self, frame_max_level, now):
        """시간 감쇠(Decay) 모델"""
        if frame_max_level >= self.global_risk_level:
            self.global_risk_level = frame_max_level
            if frame_max_level > 1:
                self.risk_expire_time = now + DECAY_STEP_SEC
        elif now > self.risk_expire_time and self.global_risk_level > 1:
            self.global_risk_level -= 1
            self.risk_expire_time = now + DECAY_STEP_SEC

    def draw_hud(self, frame, targets_meta, fps, latency_ms, video_name, current_sec, frame_idx, rc_threshold):
        h, w = frame.shape[:2]
        now = time.time()
        
        frame_max_level = max([t['risk_level'] for t in targets_meta], default=1)
        self.update_decay_system(frame_max_level, now)
        
        # 1. 바운딩 박스 & 상세 3D 텍스트 오버레이
        for t in targets_meta:
            x1, y1, x2, y2 = map(int, t['bbox'])
            color = t['color']
            dist = t['spatial']['distance_m']
            bearing = t['spatial']['bearing_deg']
            cx, cy, cz = t['spatial']['coord_3d']
            score = t['score']
            
            # Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 주 라벨 (ID, 거리, 방위각)
            label1 = f"T-{t['id']} | {dist:.1f}m | {bearing:+.1f}deg ({score*100:.1f}%)"
            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - 38), (x1 + tw1 + 6, y1 - 20), color, -1)
            cv2.putText(frame, label1, (x1 + 3, y1 - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # 보조 라벨 (3D 좌표 X, Y, Z)
            label2 = f"3D:[X:{cx:+.1f}, Y:{cy:+.1f}, Z:{cz:.1f}]m"
            (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw2 + 6, y1), (20, 20, 20), -1)
            cv2.putText(frame, label2, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Level 3 / Level 4 경보 효과
        if self.global_risk_level == 4:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)
            cw, ch = 480, 80
            cx, cy = w // 2, h // 2
            overlay = frame.copy()
            cv2.rectangle(overlay, (cx - cw//2, cy - ch//2), (cx + cw//2, cy + ch//2), (0, 0, 50), -1)
            cv2.rectangle(overlay, (cx - cw//2, cy - ch//2), (cx + cw//2, cy + ch//2), (0, 0, 255), 2)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            cv2.putText(frame, "[!] CRITICAL COLLISION WARNING", (cx - cw//2 + 25, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "EVASIVE ACTION REQUIRED", (cx - cw//2 + 95, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)
        elif self.global_risk_level == 3:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 152, 255), 5)

        # 3. 우측 상단 종합 텔레메트리 정보창
        panel_w = 370
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 10, 10), (w - 10, 230), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (w - panel_w - 10, 10), (w - 10, 230), (70, 80, 90), 1)

        lvl_names = {1: "Level 1 (Safe)", 2: "Level 2 (Caution)", 3: "Level 3 (Warning)", 4: "Level 4 (Critical)"}
        lvl_colors = {1: (0, 255, 0), 2: (0, 235, 255), 3: (0, 152, 255), 4: (0, 0, 255)}
        
        cv2.putText(frame, "IBS-PAS Aerial Analytics", (w - panel_w, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 252, 102), 2)
        cv2.putText(frame, f"File: {video_name} | Frame #{frame_idx}", (w - panel_w, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        cv2.putText(frame, f"Timestamp: {current_sec:.2f}s | FPS: {fps:.1f}", (w - panel_w, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1)
        cv2.putText(frame, f"AI Latency: {latency_ms:.1f}ms | Targets: {len(targets_meta)}", (w - panel_w, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 200), 1)
        
        # 위험도 표시
        curr_lvl = self.global_risk_level
        cv2.putText(frame, "Threat Status:", (w - panel_w, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(frame, lvl_names[curr_lvl], (w - panel_w + 110, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, lvl_colors[curr_lvl], 2)
        
        # 게이지 바
        min_dist = min([t['spatial']['distance_m'] for t in targets_meta], default=1000.0)
        gauge_ratio = max(0.0, min(1.0, (1000.0 - min_dist) / 1000.0))
        gx, gy, gw, gh = w - panel_w, 135, panel_w - 20, 14
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (50, 50, 50), -1)
        cv2.rectangle(frame, (gx, gy), (gx + int(gw * gauge_ratio), gy + gh), lvl_colors[curr_lvl], -1)
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (120, 120, 120), 1)
        cv2.putText(frame, "L1(Safe)  L2(Caut)  L3(Warn)  L4(Crit)", (gx, gy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
        
        # 1순위 위협 표적의 3D 좌표 요약
        if targets_meta:
            primary = min(targets_meta, key=lambda x: x['spatial']['distance_m'])
            px, py, pz = primary['spatial']['coord_3d']
            cv2.putText(frame, f"Primary Threat 3D Rel-Pos:", (w - panel_w, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            cv2.putText(frame, f"X:{px:+.1f}m, Y:{py:+.1f}m, Z:{pz:.1f}m", (w - panel_w, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 4. 좌측 하단 2D 전방 레이더 (130° FOV)
        self.draw_radar(frame, targets_meta, rc_threshold)
        return frame

    def draw_radar(self, frame, targets_meta, rc_threshold):
        h, w = frame.shape[:2]
        rw, rh = 220, 110
        rx, ry = 20, h - rh - 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 20, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)
        
        origin_x = rx + rw // 2
        origin_y = ry + rh
        max_r = int(rh * 0.95)
        
        for scale in [0.25, 0.5, 0.75, 1.0]:
            r_px = int(max_r * scale)
            cv2.ellipse(frame, (origin_x, origin_y), (r_px, r_px), 0, 180, 360, (0, 80, 0), 1)
            
        rc_r_px = int(max_r * (rc_threshold / 1000.0))
        cv2.ellipse(frame, (origin_x, origin_y), (rc_r_px, rc_r_px), 0, 180, 360, (0, 0, 100), -1)
        
        for t in targets_meta:
            dist = t['spatial']['distance_m']
            bearing = t['spatial']['bearing_deg']
            if dist > 1000.0:
                continue
                
            r_px = (dist / 1000.0) * max_r
            angle_rad = math.radians(-90.0 + bearing)
            px = int(origin_x + r_px * math.cos(angle_rad))
            py = int(origin_y + r_px * math.sin(angle_rad))
            
            color = (0, 0, 255) if dist <= rc_threshold else ((0, 152, 255) if dist <= 500 else (0, 235, 255))
            cv2.circle(frame, (px, py), 4, color, -1)

# ==============================================================================
# 6. 메인 헤드리스 분석 및 캡처 저장 루프
# ==============================================================================
def main():
    target_videos, args = get_target_video_files()

    if not target_videos:
        print("[종료] 분석할 영상 파일이 존재하지 않습니다.")
        return

    if not os.path.exists(args.model):
        print(f"[오류] ONNX 모델 파일을 찾을 수 없습니다: {args.model}")
        return

    # 결과 저장 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)

    print("=" * 90)
    print(" [IBS-PAS] 3D Collision Avoidance Headless Analytics & Frame Capturer")
    print("=" * 90)
    print(f"* 분석 대상 영상: {len(target_videos)}개")
    print(f"* ONNX Model: {args.model}")
    print(f"* 캡처 저장 경로: {os.path.abspath(args.output)}")
    print(f"* Horizontal FOV: {CAMERA_FOV_DEG}° | Optical Zoom: {args.zoom}x")
    print(f"* Flight State: {FLIGHT_STATE} (Rc = {192.0 if FLIGHT_STATE == 'CRUISE' else 220.0}m)")
    print("=" * 90 + "\n")

    detector = YOLO11Detector(args.model)
    visualizer = CockpitHUDVisualizer()
    rc_threshold = 192.0 if FLIGHT_STATE == 'CRUISE' else 220.0

    total_saved_images = 0

    for v_idx, video_path in enumerate(target_videos, start=1):
        if not os.path.exists(video_path):
            print(f"[경고] 파일을 찾을 수 없습니다: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        video_stem = os.path.splitext(video_name)[0]
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f">> [{v_idx}/{len(target_videos)}] 분석 시작: {video_name} (총 {total_frames} 프레임, {video_fps:.1f} FPS)")
        print("-" * 120)
        print(f"{'시간(Time)':^10} | {'프레임':^8} | {'표적ID':^6} | {'신뢰도':^7} | {'3D 상대위치 (X, Y, Z) [m]':^28} | {'거리(m)':^8} | {'방위각':^8} | {'위험도 판정':^18}")
        print("-" * 120)

        frame_idx = 0
        saved_for_this_video = 0

        while cap.isOpened():
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            current_sec = frame_idx / video_fps
            img_h, img_w = frame.shape[:2]

            # 1. 딥러닝 추론
            detections = detector.infer(frame, conf_thresh=args.conf)
            latency_ms = (time.time() - t_start) * 1000.0

            # 2. 3D 좌표 및 위험도 메타데이터 산출
            targets_meta = []
            for tid, det in enumerate(detections, start=1):
                x1, y1, x2, y2 = det['bbox']
                score = det['score']
                
                spatial = calculate_3d_spatial_data(
                    x1, y1, x2, y2, img_w, img_h, 
                    zoom=args.zoom, fov_deg=CAMERA_FOV_DEG
                )
                risk_level, risk_desc, color = evaluate_risk_level(spatial['distance_m'], rc_threshold)
                
                targets_meta.append({
                    "id": tid,
                    "bbox": det['bbox'],
                    "score": score,
                    "spatial": spatial,
                    "risk_level": risk_level,
                    "risk_desc": risk_desc,
                    "color": color
                })

                # 콘솔 텔레메트리 출력
                cx, cy, cz = spatial['coord_3d']
                print(f"[{current_sec:6.2f}s]   | #{frame_idx:<6} | T-{tid:<3} | {score*100:5.1f}% | "
                      f"X:{cx:+6.1f}m, Y:{cy:+5.1f}m, Z:{cz:6.1f}m | "
                      f"{spatial['distance_m']:6.1f}m | {spatial['bearing_deg']:+6.1f}° | "
                      f"{risk_desc}")

            # 3. 캡처 이미지 저장 조건 판정 (표적이 있거나 --save-all 옵션 활성화 시)
            should_save = len(targets_meta) > 0 or args.save_all

            if should_save:
                # 정보가 오버레이된 HUD 프레임 합성
                hud_frame = visualizer.draw_hud(
                    frame.copy(), targets_meta, video_fps, latency_ms, 
                    video_name, current_sec, frame_idx, rc_threshold
                )
                
                # 파일명 규격: {비디오이름}_f{프레임번호}_{시간초}s_L{위험도단계}.jpg
                max_lvl = max([t['risk_level'] for t in targets_meta], default=1)
                img_filename = f"{video_stem}_f{frame_idx:05d}_{current_sec:.2f}s_L{max_lvl}.jpg"
                save_path = os.path.join(args.output, img_filename)
                
                cv2.imwrite(save_path, hud_frame)
                saved_for_this_video += 1
                total_saved_images += 1

            frame_idx += 1

        cap.release()
        print(f">> [{video_name}] 분석 완료 -> {saved_for_this_video}장의 분석 결과 캡처 이미지 저장됨.\n")

    print("=" * 90)
    print(f">> [전체 완료] 총 {total_saved_images}장의 분석 캡처 파일이 '{os.path.abspath(args.output)}' 에 저장되었습니다.")
    print("=" * 90)

if __name__ == "__main__":
    main()