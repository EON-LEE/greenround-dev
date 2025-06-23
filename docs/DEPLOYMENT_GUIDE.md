# Golf 3D Analyzer - Google Cloud 배포 가이드 🚀

## 📋 목차
1. [빠른 시작 (5분 완성)](#빠른-시작-5분-완성)
2. [환경 설정 (최초 1회만)](#환경-설정-최초-1회만)
3. [배포 스크립트 사용법](#배포-스크립트-사용법)
4. [문제 해결](#문제-해결)
5. [모니터링 및 관리](#모니터링-및-관리)

---

## 🚀 빠른 시작 (5분 완성)

> **이미 GCP 환경이 설정된 경우 바로 배포하세요!**

### 1단계: 사전 확인 ✅
```bash
# 필수 도구 설치 확인
docker --version
gcloud --version

# Google Cloud 로그인 확인
gcloud auth list
```

### 2단계: 배포 실행 🚀
```bash
# 배포 스크립트 실행 권한 부여
chmod +x deploy_to_gcp.sh

# 전체 배포 (빌드 + 배포)
./deploy_to_gcp.sh
```

### 3단계: 확인 ✨
```bash
# 배포 상태 확인
./deploy_to_gcp.sh --status
```

**완료!** 🎉 약 3-5분 후 API가 Google Cloud Run에서 실행됩니다.

---

## ⚙️ 환경 설정 (최초 1회만)

> **이미 GCP 콘솔에서 설정했다면 이 섹션은 건너뛰세요!**

### 옵션 1: 자동 설정 스크립트 (권장)
```bash
# 환경 설정 스크립트 실행
chmod +x setup_gcp_environment.sh
./setup_gcp_environment.sh
```

**스크립트가 자동으로 생성하는 것들:**
- 필수 API 활성화 (Artifact Registry, Cloud Run, Storage 등)
- 서비스 계정 `golf-analyzer-sa` 생성 및 권한 부여
- Artifact Registry 저장소 생성
- GCS 버킷 생성
- 환경 변수 파일 `.env` 생성
- 서비스 계정 키 `gcs-credentials.json` 생성

### 옵션 2: 수동 설정 (이미 콘솔에서 설정한 경우)

#### 2-1. 필수 API 활성화 확인
```bash
gcloud services list --enabled --filter="name:(artifactregistry.googleapis.com OR run.googleapis.com OR cloudbuild.googleapis.com OR storage.googleapis.com)"
```

#### 2-2. 환경 변수 파일 생성
`.env` 파일을 생성하고 프로젝트 정보를 입력:
```bash
# Google Cloud 설정
GCP_PROJECT_ID=your-project-id
GCP_REGION=asia-northeast3
GCP_SERVICE_NAME=golf-analyzer-backend
GCP_REPOSITORY=golf-analyzer
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=gcs-credentials.json

# 애플리케이션 설정
ENVIRONMENT=production
```

#### 2-3. 서비스 계정 키 다운로드
Google Cloud Console에서 서비스 계정 키를 다운로드하여 `gcs-credentials.json`으로 저장

---

## 📜 배포 스크립트 사용법

### 기본 배포 명령어
```bash
# 전체 배포 (빌드 + 배포)
./deploy_to_gcp.sh

# 특정 버전으로 배포
./deploy_to_gcp.sh v1.2.0

# 빌드만 수행
./deploy_to_gcp.sh --build-only

# 기존 이미지로 재배포만
./deploy_to_gcp.sh --deploy-only
```

### 모니터링 명령어
```bash
# 서비스 상태 확인
./deploy_to_gcp.sh --status

# 실시간 로그 보기
./deploy_to_gcp.sh --logs

# 환경 설정 확인
./deploy_to_gcp.sh --check

# 초기 설정 수행
./deploy_to_gcp.sh --setup
```

### 배포 과정 상세
1. **Docker 인증 설정**: Artifact Registry 접근 권한 설정
2. **이미지 빌드**: Linux/AMD64 플랫폼으로 크로스 빌드
3. **이미지 푸시**: Artifact Registry에 업로드
4. **Cloud Run 배포**: 서비스 생성/업데이트
5. **상태 확인**: API 응답 테스트

---

## 🔍 문제 해결

### 자주 발생하는 오류들

#### 1. OpenCV 설치 오류
```bash
# 해결 방법
pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y
pip install opencv-contrib-python==4.8.1.78
```

#### 2. Docker 인증 오류
```bash
# Docker 설정 초기화
rm ~/.docker/config.json
gcloud auth configure-docker asia-northeast3-docker.pkg.dev
```

#### 3. GCS 인증 오류 (로컬 환경)
```bash
# 서비스 계정 키 설정
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcs-credentials.json"

# 또는 Application Default Credentials 설정
gcloud auth application-default login
```

#### 4. 메모리 부족 오류
```bash
# Cloud Run 메모리 증설
gcloud run services update golf-analyzer-backend \
    --region=asia-northeast3 \
    --memory=4Gi \
    --cpu=2
```

#### 5. 아키텍처 호환성 문제 (M1/M2 Mac)
```bash
# Docker Buildx 설정
docker buildx create --use --name multiarch
docker buildx build --platform linux/amd64 --push -t IMAGE_NAME .
```

### 배포 실패 시 체크리스트
- [ ] Google Cloud CLI 로그인 상태 확인
- [ ] Docker Desktop 실행 상태 확인
- [ ] 프로젝트 ID 정확성 확인
- [ ] 필수 API 활성화 상태 확인
- [ ] 서비스 계정 권한 확인
- [ ] `.env` 파일 존재 및 내용 확인

---

## 🌐 배포 후 확인사항

### API 엔드포인트 테스트
```bash
# 서비스 URL 가져오기
SERVICE_URL=$(gcloud run services describe golf-analyzer-backend --region=asia-northeast3 --format='value(status.url)')

# API 문서 확인
curl -s "$SERVICE_URL/docs"

# 헬스 체크
curl -s "$SERVICE_URL/api/health"

# 시스템 정보
curl -s "$SERVICE_URL/api/info"
```

### 주요 접속 URL
- **API 문서**: `https://서비스-URL/docs`
- **OpenAPI 스키마**: `https://서비스-URL/openapi.json`
- **헬스 체크**: `https://서비스-URL/api/health`
- **시스템 정보**: `https://서비스-URL/api/info`

---

## 📊 모니터링 및 관리

### 실시간 로그 모니터링
```bash
# 배포 스크립트로 로그 보기
./deploy_to_gcp.sh --logs

# 직접 gcloud 명령어 사용
gcloud logs tail --follow \
    --resource-labels=service_name=golf-analyzer-backend \
    --resource-labels=location=asia-northeast3
```

### 서비스 업데이트
```bash
# 새 버전 배포
./deploy_to_gcp.sh v2.0.0

# 트래픽 분할 (카나리 배포)
gcloud run services update-traffic golf-analyzer-backend \
    --region=asia-northeast3 \
    --to-revisions=REVISION-1=50,REVISION-2=50
```

### 스케일링 설정
```bash
# 최대/최소 인스턴스 수 조정
gcloud run services update golf-analyzer-backend \
    --region=asia-northeast3 \
    --max-instances=20 \
    --min-instances=1 \
    --concurrency=1000
```

### 비용 최적화
- **인스턴스 수 제한**: `--max-instances` 설정
- **CPU/메모리 최적화**: 필요에 따라 리소스 조정
- **리비전 정리**: 오래된 리비전 삭제
- **청구 알람 설정**: Google Cloud Console에서 예산 알람 설정

---

## 🛠️ 개발 워크플로우

### 로컬 개발 환경
```bash
# 로컬에서 백엔드 실행
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 배포 워크플로우
1. **코드 수정** → Git 커밋
2. **로컬 테스트** → 기능 확인
3. **배포 실행** → `./deploy_to_gcp.sh`
4. **배포 확인** → API 테스트
5. **모니터링** → 로그 및 성능 확인

### 버전 관리
```bash
# Git 태그로 버전 관리
git tag v1.2.0
git push origin v1.2.0

# 특정 버전으로 배포
./deploy_to_gcp.sh v1.2.0
```

---

## 🔗 유용한 링크

- [Google Cloud Console](https://console.cloud.google.com)
- [Cloud Run 문서](https://cloud.google.com/run/docs)
- [Artifact Registry 문서](https://cloud.google.com/artifact-registry/docs)
- [Docker Buildx 문서](https://docs.docker.com/buildx/)

---

## 💡 팁 & 베스트 프랙티스

### 개발 팁
1. **로컬 개발**: GCS 인증 없이도 로컬 파일로 동작
2. **빠른 테스트**: `--deploy-only`로 빌드 시간 단축
3. **로그 확인**: 문제 발생 시 실시간 로그로 디버깅

### 보안 팁
1. **서비스 계정 키**: `gcs-credentials.json` 파일 보안 관리
2. **환경 변수**: 민감한 정보는 환경 변수로 관리
3. **권한 최소화**: 필요한 권한만 부여

### 성능 팁
1. **리소스 모니터링**: CPU/메모리 사용량 주기적 확인
2. **캐시 활용**: Docker 빌드 캐시 최적화
3. **인스턴스 관리**: 트래픽에 따른 스케일링 설정

---

**📞 지원이 필요하신가요?**
- 문제 해결: 위 문제 해결 섹션 참조
- 추가 도움: GitHub Issues 등록 