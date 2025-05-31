# Golf Swing 3D Analyzer

골프 스윙 분석을 위한 FastAPI 기반 백엔드 서비스입니다.

## 프로젝트 구조

```
.
├── src/
│   ├── app.py              # FastAPI 애플리케이션 메인
│   ├── pose_estimation.py  # 포즈 추정 모듈
│   ├── swing_analyzer.py   # 스윙 분석 모듈
│   └── video_processor.py  # 비디오 처리 모듈
├── requirements.txt        # 의존성 패키지
└── README.md              # 프로젝트 문서
```

## 설치 방법

1. Python 3.8 이상이 필요합니다.

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

## 실행 방법

1. 서버 실행:
```bash
cd src
uvicorn app:app --reload
#uvicorn src.main:app --reload --log-level debug
```

2. API 문서 확인:
- http://localhost:8000/docs

## API 엔드포인트

### 1. 비디오 업로드
- **엔드포인트**: `/api/video/upload`
- **메소드**: POST
- **입력**: 비디오 파일 (mp4, avi, mov)
- **출력**: 
  ```json
  {
    "message": "비디오가 성공적으로 업로드되었습니다.",
    "file_path": "임시_파일_경로",
    "video_id": "고유_비디오_ID"
  }
  ```

### 2. 비디오 분석
- **엔드포인트**: `/api/analysis/analysis/{video_id}`
- **메소드**: GET
- **입력**: 비디오 ID (URL 파라미터)
- **출력**: 
  ```json
  {
    "message": "분석이 완료되었습니다.",
    "frames": [...],  // 프레임별 분석 데이터
    "metrics": {...}  // 스윙 메트릭스
  }
  ```

### 3. 비디오 스트리밍
- **엔드포인트**: `/api/video/{video_id}`
- **메소드**: GET
- **입력**: 비디오 ID (URL 파라미터)
- **출력**: 비디오 파일 스트림 (video/mp4) 