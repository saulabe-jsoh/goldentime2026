# IBS-PAS (Robust Edge Vision & Cockpit HUD)

**IBS-PAS (Intelligent Bird & Drone Collision Prevention Air Safety System)**는 비협조 표적(조류 및 미확인 드론)을 실시간으로 조기 탐지하고, 
3D 상대 공간 좌표 및 방위각 기반 거리 판단을 통해 능동적 공중 충돌 예방 의사결정을 지원하는 엣지 비전 AI 시스템입니다.

---

## 📌 주요 특징

* **YOLO11 ONNX 엣지 비전 추론**: 640×640 해상도 기반 초저지연 객체 탐지 및 GPU End-to-End NMS 지원
* **3D 공간 좌표 및 방위각(Bearing-Only) 모델링**: 130° FOV와 15x 광학 줌을 연계한 $X$(수평 편차), $Y$(고도 편차), $Z$(전방 거리) 텔레메트리 
실시간 산출
* **4단계 위험도 판단 체계**:
  * **Level 1 (Safe)**: 거리 > 750m
  * **Level 2 (Caution)**: 500m ~ 750m
  * **Level 3 (Warning)**: $R_c$ ~ 500m (주황색 점멸 테두리)
  * **Level 4 (Critical)**: 거리 $\le R_c$ (적색 점멸 테두리 + 중앙 경고 삼각형 `[!] CRITICAL WARNING`)
* **상황 인식 연속성을 위한 10초 점진적 시간 감쇠(Decay)**: 3.3초 간격 단계적 위험도 감쇠로 조종사 오판 방지
* **실시간 조종석 풀 HUD 및 130° 전방 레이더 시각화**: 위험도 게이지 바, AI Latency, FPS, 2D 극좌표 레이더 투영
* **유연한 파일 입력 인터페이스**: GUI 파일 탐색기, CLI 인자 지정, 기본 시나리오(`test0.mp4` ~ `test8.mp4`) 순차 자동 실행 지원

---
1. 구조

├── main.py                     # IBS-PAS 비전 추론 및 HUD 렌더링 메인 스크립트
├── ibs-n-final.html             # 브라우저 엣지 구동용 WebGL/WASM 시뮬레이터
├── yolo11n.onnx                # YOLO11 경량화 ONNX 모델 파일
├── test0.mp4 ~ test8.mp4       # 시나리오별 실증 테스트 영상 (0~8)
├── requirements.txt            # Python 의존성 패키지 목록
└── README.md                   # 프로젝트 매뉴얼

2. 필수 라이브러리 설치

pip install -r requirements.txt

***** 참고 (GPU 가속 사용 시):
NVIDIA GPU(CUDA) 가속을 사용하려면 시스템 CUDA 버전에 맞는 onnxruntime-gpu를 설치해야 합니다:

pip uninstall onnxruntime
pip install onnxruntime-gpu


3. 실행

python test_goldentime.py -i ./test2.mp4 --conf 0.30 --zoom 30.0 --model ./yolo11n.onnx

폴더 분석
python test_goldentime.py --input ./test_dataset/


================================================================================
 [IBS-PAS] Python Edge Vision & 3D Aerial Collision Avoidance System
================================================================================
* 분석 대상 파일 수: 1개
   [1] ./test2.mp4
* Model: ./yolo11n.onnx
* Horizontal FOV: 130.0° | Optical Zoom: 15.0x
* Flight State: CRUISE (Rc = 192.0m)
================================================================================

>> [1/1] 분석 시작: test2.mp4 (총 450 프레임, 30.0 FPS)
-------------------------------------------------------------------------------------------------------------------
 시간(Time)  | 표적ID |  신뢰도  |    3D 상대위치 (X, Y, Z) [m]     |  거리(m)  |  방위각  |      위험도 판정      
-------------------------------------------------------------------------------------------------------------------
[  1.20s]   | T-1    |  89.4% | X: -12.4m, Y: +4.1m, Z: 184.2m |   184.6m |   -3.8° | Level 4 (Critical)
[  1.23s]   | T-1    |  91.2% | X: -10.1m, Y: +3.8m, Z: 172.0m |   172.3m |   -3.3° | Level 4 (Critical)

###########
1. 파이썬 간이 웹서버로 실행 (가장 추천 & 1초 해결)
브라우저에서 file:///C:/... 형태로 HTML 파일을 직접 열면 동일 출처(Origin)가 없어 무조건 캔버스 오염 에러가 발생합니다. 로컬 웹 서버로 열면 이 문제가 완전히 해결됩니다.

HTML 및 영상 파일이 있는 폴더에서 터미널(명령 프롬프트)을 엽니다.

아래 명령어를 실행합니다:

python -m http.server 8000

브라우저 주소창에 다음 주소로 접속합니다:

http://localhost:8000/ibs-local.html