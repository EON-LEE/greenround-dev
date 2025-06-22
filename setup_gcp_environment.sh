#!/bin/bash

# =============================================================================
# Greenround - Google Cloud 환경 초기 설정 스크립트
# =============================================================================

set -e

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 사용자로부터 입력 받기
get_user_input() {
    echo "=========================================="
    echo "  Greenround - GCP 환경 설정"
    echo "=========================================="
    echo ""
    
    # 새 프로젝트 생성 여부 선택
    echo "프로젝트 설정 방법을 선택하세요:"
    echo "1. 새 프로젝트 자동 생성 (추천)"
    echo "2. 기존 프로젝트 사용"
    echo ""
    read -p "선택하세요 (1-2, 기본값: 1): " PROJECT_CHOICE
    
    if [ "$PROJECT_CHOICE" = "2" ]; then
        # 기존 프로젝트 사용
        read -p "기존 GCP 프로젝트 ID를 입력하세요: " PROJECT_ID
        if [ -z "$PROJECT_ID" ]; then
            log_error "프로젝트 ID는 필수입니다."
            exit 1
        fi
        CREATE_NEW_PROJECT=false
    else
        # 새 프로젝트 자동 생성 (랜덤 서픽스)
        PROJECT_SUFFIX=$(date +%s | tail -c 6)
        PROJECT_ID="greenround-${PROJECT_SUFFIX}"
        log_info "새 프로젝트 ID 자동 생성: $PROJECT_ID"
        CREATE_NEW_PROJECT=true
    fi
    
    # 리소스명 랜덤 서픽스 생성
    RESOURCE_SUFFIX=$(openssl rand -hex 4 2>/dev/null || echo $(date +%s | tail -c 8))
    
    # Firestore 데이터베이스 ID 미리 생성
    FIRESTORE_DATABASE_ID="greenround-db-${RESOURCE_SUFFIX}"
    
    # 리전 선택
    echo ""
    echo "사용 가능한 리전:"
    echo "1. asia-northeast3 (서울)"
    echo "2. asia-northeast1 (도쿄)"
    echo "3. us-central1 (아이오와)"
    echo "4. europe-west1 (벨기에)"
    echo ""
    read -p "리전을 선택하세요 (1-4, 기본값: 1): " REGION_CHOICE
    
    case $REGION_CHOICE in
        1|"") REGION="asia-northeast3" ;;
        2) REGION="asia-northeast1" ;;
        3) REGION="us-central1" ;;
        4) REGION="europe-west1" ;;
        *) 
            log_warning "잘못된 선택입니다. 기본값(서울)을 사용합니다."
            REGION="asia-northeast3"
            ;;
    esac
    
    # 서비스 이름 (랜덤 서픽스 포함)
    DEFAULT_SERVICE_NAME="greenround-backend-${RESOURCE_SUFFIX}"
    read -p "서비스 이름을 입력하세요 (기본값: $DEFAULT_SERVICE_NAME): " SERVICE_NAME
    SERVICE_NAME=${SERVICE_NAME:-$DEFAULT_SERVICE_NAME}
    
    # GCS 버킷 이름 (랜덤 서픽스 포함)
    DEFAULT_BUCKET_NAME="greenround-storage-${RESOURCE_SUFFIX}"
    read -p "GCS 버킷 이름을 입력하세요 (기본값: $DEFAULT_BUCKET_NAME): " BUCKET_NAME
    BUCKET_NAME=${BUCKET_NAME:-$DEFAULT_BUCKET_NAME}
}

# 새 프로젝트 생성 (선택적)
create_new_project() {
    if [ "$CREATE_NEW_PROJECT" = "true" ]; then
        log_info "새 GCP 프로젝트 생성 중: $PROJECT_ID"
        
        # 프로젝트 생성
        if gcloud projects create "$PROJECT_ID" --name="Greenround Backend" 2>/dev/null; then
            log_success "새 프로젝트가 생성되었습니다: $PROJECT_ID"
        else
            log_warning "프로젝트 생성 실패 또는 이미 존재함. 기존 프로젝트 사용을 시도합니다."
        fi
        
        # 결제 계정 자동 연결
        log_info "프로젝트에 결제 계정을 연결합니다..."
        
        # 사용 가능한 결제 계정 조회
        BILLING_ACCOUNTS=$(gcloud billing accounts list --format="value(name)" --filter="open=true" 2>/dev/null || echo "")
        
        if [ -z "$BILLING_ACCOUNTS" ]; then
            log_warning "사용 가능한 결제 계정을 찾을 수 없습니다."
            log_info "Google Cloud Console에서 결제 계정을 연결하거나, 다음 명령어를 사용하세요:"
            log_info "gcloud billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID"
            echo ""
            read -p "결제 계정 연결을 완료했다면 Enter를 눌러 계속하세요..."
        else
            # 결제 계정이 하나만 있는 경우 자동 연결
            BILLING_COUNT=$(echo "$BILLING_ACCOUNTS" | wc -l)
            
            if [ "$BILLING_COUNT" -eq 1 ]; then
                BILLING_ACCOUNT_ID=$(echo "$BILLING_ACCOUNTS" | head -1)
                log_info "결제 계정 자동 연결 중: $BILLING_ACCOUNT_ID"
                
                if gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID" 2>/dev/null; then
                    log_success "결제 계정이 자동으로 연결되었습니다: $BILLING_ACCOUNT_ID"
                else
                    log_warning "자동 연결 실패. 수동으로 연결해주세요."
                    echo ""
                    read -p "결제 계정 연결을 완료했다면 Enter를 눌러 계속하세요..."
                fi
            else
                # 여러 결제 계정이 있는 경우 선택
                log_info "사용 가능한 결제 계정들:"
                echo "$BILLING_ACCOUNTS" | nl -w2 -s'. '
                echo ""
                read -p "사용할 결제 계정 번호를 선택하세요 (1-$BILLING_COUNT): " BILLING_CHOICE
                
                if [[ "$BILLING_CHOICE" =~ ^[0-9]+$ ]] && [ "$BILLING_CHOICE" -ge 1 ] && [ "$BILLING_CHOICE" -le "$BILLING_COUNT" ]; then
                    SELECTED_BILLING_ACCOUNT=$(echo "$BILLING_ACCOUNTS" | sed -n "${BILLING_CHOICE}p")
                    log_info "선택된 결제 계정으로 연결 중: $SELECTED_BILLING_ACCOUNT"
                    
                    if gcloud billing projects link "$PROJECT_ID" --billing-account="$SELECTED_BILLING_ACCOUNT" 2>/dev/null; then
                        log_success "결제 계정이 연결되었습니다: $SELECTED_BILLING_ACCOUNT"
                    else
                        log_error "결제 계정 연결에 실패했습니다."
                        exit 1
                    fi
                else
                    log_error "잘못된 선택입니다."
                    exit 1
                fi
            fi
        fi
    fi
}

# 프로젝트 설정
setup_project() {
    log_info "프로젝트 설정 중..."
    
    # 프로젝트 존재 확인
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_error "프로젝트 '$PROJECT_ID'를 찾을 수 없습니다."
        if [ "$CREATE_NEW_PROJECT" = "true" ]; then
            log_error "새 프로젝트 생성에 실패했습니다."
        else
            log_info "올바른 프로젝트 ID를 입력했는지 확인하세요."
        fi
        exit 1
    fi
    
    # 프로젝트 설정
    gcloud config set project "$PROJECT_ID"
    log_success "프로젝트가 설정되었습니다: $PROJECT_ID"
}

# 필수 API 활성화
enable_required_apis() {
    log_info "필수 API 활성화 중..."
    
    apis=(
        "artifactregistry.googleapis.com"
        "run.googleapis.com"
        "cloudbuild.googleapis.com"
        "storage.googleapis.com"
        "iam.googleapis.com"
        "firestore.googleapis.com"
        "compute.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "API 활성화: $api"
        gcloud services enable "$api" --project="$PROJECT_ID"
    done
    
    log_success "모든 API가 활성화되었습니다."
}

# 서비스 계정 생성
create_service_account() {
    log_info "서비스 계정 생성 중..."
    
    SA_NAME="greenround-sa-${RESOURCE_SUFFIX}"
    SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    # 서비스 계정이 이미 존재하는지 확인
    if gcloud iam service-accounts describe "$SA_EMAIL" &> /dev/null; then
        log_warning "서비스 계정이 이미 존재합니다: $SA_EMAIL"
    else
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="greenround Service Account" \
            --description="greenround를 위한 서비스 계정"
        log_success "서비스 계정이 생성되었습니다: $SA_EMAIL"
    fi
    
    # 권한 부여
    log_info "서비스 계정에 권한 부여 중..."
    roles=(
        "roles/storage.admin"                    # GCS 버킷 관리
        "roles/run.admin"                        # Cloud Run 서비스 관리 (developer → admin으로 변경)
        "roles/artifactregistry.admin"           # Artifact Registry 관리 (writer → admin으로 변경)
        "roles/datastore.user"                   # Firestore 데이터베이스 사용
        "roles/serviceusage.serviceUsageAdmin"   # API 활성화 권한 (중요!)
        "roles/compute.admin"                    # Compute Engine 리소스 관리
        "roles/cloudbuild.builds.editor"         # Cloud Build 관리
        "roles/iam.serviceAccountUser"           # 서비스 계정 사용 권한
        "roles/logging.logWriter"                # 로그 작성 권한
        "roles/monitoring.metricWriter"          # 모니터링 메트릭 작성
    )
    
    for role in "${roles[@]}"; do
        log_info "권한 부여: $role"
        # 기존 바인딩 확인 후 추가
        if ! gcloud projects get-iam-policy "$PROJECT_ID" --flatten="bindings[].members" --format="table(bindings.role)" --filter="bindings.members:serviceAccount:$SA_EMAIL AND bindings.role:$role" | grep -q "$role"; then
            gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:$SA_EMAIL" \
                --role="$role" \
                --condition=None 2>/dev/null || \
            gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:$SA_EMAIL" \
                --role="$role"
            log_success "권한 부여 완료: $role"
        else
            log_warning "권한이 이미 존재함: $role"
        fi
    done
    
    # 서비스 계정 키 생성
    if [ -f "gcs-credentials.json" ]; then
        log_warning "기존 서비스 계정 키 파일이 존재합니다."
        echo "기존 파일: $(grep -o '"project_id": "[^"]*"' gcs-credentials.json 2>/dev/null || echo '정보 없음')"
        echo "새 프로젝트: $PROJECT_ID"
        echo ""
        read -p "새 서비스 계정 키를 생성하시겠습니까? 기존 파일은 백업됩니다. (y/N): " replace_key
        
        if [[ "$replace_key" =~ ^[Yy]$ ]]; then
            mv gcs-credentials.json "gcs-credentials.json.backup.$(date +%Y%m%d_%H%M%S)"
            log_info "기존 파일을 백업했습니다."
        else
            log_warning "기존 서비스 계정 키를 유지합니다. 프로젝트가 다를 경우 인증 오류가 발생할 수 있습니다."
            return
        fi
    fi
    
    log_info "새 서비스 계정 키 생성 중..."
    gcloud iam service-accounts keys create gcs-credentials.json \
        --iam-account="$SA_EMAIL"
    log_success "서비스 계정 키가 생성되었습니다: gcs-credentials.json"
}

# Artifact Registry 저장소 생성
create_artifact_repository() {
    log_info "Artifact Registry 저장소 생성 중..."
    
    REPO_NAME="greenround-${RESOURCE_SUFFIX}"
    
    if gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &> /dev/null; then
        log_warning "Artifact Registry 저장소가 이미 존재합니다."
    else
        gcloud artifacts repositories create "$REPO_NAME" \
            --repository-format=docker \
            --location="$REGION" \
            --description="Greenround Docker Repository"
        log_success "Artifact Registry 저장소가 생성되었습니다."
    fi
}

# GCS 버킷 생성
create_storage_bucket() {
    log_info "Google Cloud Storage 버킷 생성 중..."
    
    if gsutil ls "gs://$BUCKET_NAME" &> /dev/null; then
        log_warning "GCS 버킷이 이미 존재합니다: $BUCKET_NAME"
    else
        gsutil mb -l "$REGION" "gs://$BUCKET_NAME"
        log_success "GCS 버킷이 생성되었습니다: $BUCKET_NAME"
    fi
    
    # 버킷 권한 설정
    gsutil iam ch "serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin" "gs://$BUCKET_NAME"
}

# 환경 변수 파일 생성
create_env_file() {
    log_info "환경 변수 파일 생성 중..."
    
    cat > .env << EOF
# =============================================================================
# Greenround 환경 변수 설정 파일
# =============================================================================
# 이 파일은 로컬 개발과 프로덕션 배포 모두에서 사용됩니다.
# Docker 빌드 시 컨테이너로 복사되어 환경 변수로 로드됩니다.

# Google Cloud 설정
GCP_PROJECT_ID=$PROJECT_ID
GCP_REGION=$REGION
GCP_SERVICE_NAME=$SERVICE_NAME
GCP_REPOSITORY=$REPO_NAME
GCS_BUCKET_NAME=$BUCKET_NAME
GOOGLE_APPLICATION_CREDENTIALS=gcs-credentials.json

# 애플리케이션 설정
ENVIRONMENT=production

# Firestore 설정 (상태 영구 저장)
ENABLE_FIRESTORE_SYNC=true
FIRESTORE_PROJECT_ID=$PROJECT_ID
FIRESTORE_DATABASE_ID=$FIRESTORE_DATABASE_ID

# 서비스 URL (배포 후 자동 업데이트됨)
# SERVICE_BASE_URL=https://your-service-url.run.app
EOF
    
    log_success "환경 변수 파일이 생성되었습니다: .env"
}

# Firestore 데이터베이스 설정
setup_firestore() {
    log_info "Firestore 데이터베이스 설정 중..."
    
    # Firestore 데이터베이스 생성 (Native 모드, 고유 ID 지정)
    if ! gcloud firestore databases describe --database="$FIRESTORE_DATABASE_ID" --location="$REGION" &> /dev/null; then
        log_info "Firestore 데이터베이스 생성 중: $FIRESTORE_DATABASE_ID"
        gcloud firestore databases create \
            --database="$FIRESTORE_DATABASE_ID" \
            --location="$REGION" \
            --type=firestore-native
        log_success "Firestore 데이터베이스가 생성되었습니다: $FIRESTORE_DATABASE_ID"
    else
        log_success "Firestore 데이터베이스가 이미 존재합니다: $FIRESTORE_DATABASE_ID"
    fi
}

# Docker 인증 설정
setup_docker_auth() {
    log_info "Docker 인증 설정 중..."
    gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
    log_success "Docker 인증이 설정되었습니다."
}

# 설정 요약 출력
print_summary() {
    echo ""
    echo "=========================================="
    echo "          설정 완료 요약"
    echo "=========================================="
    echo "프로젝트 ID: $PROJECT_ID"
    
    # 연결된 결제 계정 정보 표시
    LINKED_BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || echo "정보 없음")
    if [ "$LINKED_BILLING" != "정보 없음" ] && [ ! -z "$LINKED_BILLING" ]; then
        echo "결제 계정: $LINKED_BILLING"
    else
        echo "결제 계정: 연결되지 않음 (수동 연결 필요)"
    fi
    
    echo "리전: $REGION"
    echo "서비스 이름: $SERVICE_NAME"
    echo "GCS 버킷: $BUCKET_NAME"
    echo "서비스 계정: $SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    echo "Artifact Registry: $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"
    echo ""
    echo "생성된 파일:"
    echo "- .env (통합 환경 변수 설정 - 로컬/프로덕션 공용)"
    echo "- gcs-credentials.json (서비스 계정 키)"
    echo ""
    echo "🔥 Firestore 기능:"
    echo "- Firestore Native 데이터베이스 생성 완료: $FIRESTORE_DATABASE_ID"
    echo "- 작업 상태 영구 저장 및 복구 기능 활성화"
    echo "- Cloud Run 재시작 시 상태 손실 방지"
    echo ""
    echo "다음 단계:"
    echo "1. 로컬 테스트: ENABLE_FIRESTORE_SYNC=true python backend/main.py"
    echo "2. 배포 스크립트 실행: ./deploy_to_gcp.sh"
    echo "3. Firestore 상태 확인: curl https://서비스URL/api/firestore/status"
    echo "=========================================="
}

# 메인 함수
main() {
    # 필수 도구 확인
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud CLI가 설치되지 않았습니다."
        log_info "설치 방법: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # GCP 로그인 확인
    if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" | grep -q "@"; then
        log_error "Google Cloud에 로그인이 필요합니다."
        log_info "다음 명령어를 실행하세요: gcloud auth login"
        exit 1
    fi
    
    # 사용자 입력 받기
    get_user_input
    
    # 설정 진행
    create_new_project
    setup_project
    enable_required_apis
    create_service_account
    create_artifact_repository
    create_storage_bucket
    setup_firestore
    create_env_file
    setup_docker_auth
    
    # 완료 요약
    print_summary
    
    log_success "Greenround Google Cloud 환경 설정이 완료되었습니다!"
}

# 스크립트 실행
main "$@" 