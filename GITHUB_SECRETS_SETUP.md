# GitHub Secrets 설정 가이드 🔐

GitHub Actions에서 GCP 배포를 위해 필요한 Secrets을 설정하는 방법입니다.

## 📋 필요한 Secrets 목록

| Secret 이름 | 설명 | 예시 |
|-------------|------|------|
| `GCP_SA_KEY` | GCP 서비스 계정 JSON 키 | `{"type": "service_account", ...}` |
| `GCP_PROJECT_ID` | GCP 프로젝트 ID | `greenround-prod-12345` |

---

## 🔧 1단계: GCP 서비스 계정 생성

### 1. GCP Console에서 서비스 계정 생성
```bash
# gcloud CLI로 생성 (권장)
gcloud iam service-accounts create github-actions \
    --description="GitHub Actions deployment service account" \
    --display-name="GitHub Actions"

# 서비스 계정 이메일 확인
SA_EMAIL=$(gcloud iam service-accounts list \
    --filter="displayName:GitHub Actions" \
    --format="value(email)")

echo "생성된 서비스 계정: $SA_EMAIL"
```

### 2. 필요한 권한 부여
```bash
# 프로젝트 ID 설정
PROJECT_ID="your-project-id"

# Cloud Run 관련 권한
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/run.admin"

# Artifact Registry 권한
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/artifactregistry.admin"

# Cloud Build 권한
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/cloudbuild.builds.editor"

# 서비스 활성화 권한
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/serviceusage.serviceUsageAdmin"

# IAM 권한 (서비스 계정에 권한 부여용)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/iam.serviceAccountUser"
```

### 3. JSON 키 파일 생성
```bash
# JSON 키 파일 다운로드
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=$SA_EMAIL

echo "✅ JSON 키 파일이 생성되었습니다: github-actions-key.json"
echo "⚠️  이 파일을 GitHub Secrets에 등록 후 즉시 삭제하세요!"
```

---

## 🔧 2단계: GitHub Secrets 등록

### 1. GitHub Repository → Settings → Secrets and variables → Actions

### 2. "New repository secret" 클릭하여 다음 추가:

#### `GCP_SA_KEY`
- **Name**: `GCP_SA_KEY`
- **Secret**: `github-actions-key.json` 파일의 전체 내용 복사 붙여넣기
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "github-actions@your-project.iam.gserviceaccount.com",
  "client_id": "123456789...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/github-actions%40your-project.iam.gserviceaccount.com"
}
```

#### `GCP_PROJECT_ID`
- **Name**: `GCP_PROJECT_ID`
- **Secret**: 실제 GCP 프로젝트 ID (예: `greenround-prod-12345`)

---

## 🧪 3단계: 설정 확인

### 로컬에서 테스트
```bash
# 서비스 계정 키 테스트
export GOOGLE_APPLICATION_CREDENTIALS="./github-actions-key.json"
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# 권한 확인
gcloud run services list --region=asia-northeast3
gcloud artifacts repositories list --location=asia-northeast3
```

### GitHub Actions 수동 실행
1. GitHub Repository → Actions 탭
2. "Deploy to Cloud Run" 워크플로우 선택
3. "Run workflow" → "Run workflow" 클릭
4. 실행 로그 확인

---

## 🔒 보안 고려사항

### ✅ 해야 할 것:
- JSON 키 파일을 GitHub Secrets에 등록 후 **즉시 로컬에서 삭제**
- 서비스 계정에 **최소 권한만** 부여
- 정기적으로 서비스 계정 키 로테이션

### ❌ 하지 말아야 할 것:
- JSON 키를 코드나 `.env` 파일에 커밋
- 과도한 권한 부여 (예: `roles/owner`)
- 여러 환경에서 같은 서비스 계정 사용

---

## 🎯 최종 체크리스트

- [ ] GCP 서비스 계정 생성 완료
- [ ] 필요한 IAM 권한 부여 완료
- [ ] JSON 키 파일 다운로드 완료
- [ ] GitHub Secrets `GCP_SA_KEY` 등록 완료
- [ ] GitHub Secrets `GCP_PROJECT_ID` 등록 완료
- [ ] 로컬 JSON 키 파일 삭제 완료
- [ ] GitHub Actions 수동 실행 테스트 완료

---

## 🆘 문제 해결

### 권한 오류가 발생하는 경우:
```bash
# 현재 권한 확인
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:github-actions@*"
```

### API가 비활성화된 경우:
```bash
# 필요한 API 수동 활성화
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

이제 GitHub에 코드를 푸시하면 자동으로 배포가 실행됩니다! 🚀 