# 🔥 Firestore 통합 안전 배포 계획

## 🎯 목표
기존 동작 코드를 **절대 망가뜨리지 않고** Firestore 비동기 상태 관리를 안전하게 추가

## 📋 배포 단계

### **Phase 1: 로컬 테스트 (Firestore 비활성화)**
```bash
# 1. 환경 변수 설정 (Firestore 비활성화)
export ENABLE_FIRESTORE_SYNC=false

# 2. 로컬 서버 실행
cd backend
python main.py

# 3. 기존 기능 테스트
python ../test/test_cloudrun.py
```

**검증 포인트:**
- ✅ 기존 API 모든 기능 정상 동작
- ✅ 메모리 상태 관리 정상 동작
- ✅ Firestore 관련 코드가 실행되지 않음

### **Phase 2: 로컬 테스트 (Firestore 활성화)**
```bash
# 1. Google Cloud 인증 설정
export GOOGLE_APPLICATION_CREDENTIALS="gcs-credentials.json"

# 2. 환경 변수 설정 (Firestore 활성화)
export ENABLE_FIRESTORE_SYNC=true

# 3. Firestore 연결 테스트
curl http://localhost:8000/api/firestore/status

# 4. 전체 기능 테스트
python ../test/test_cloudrun.py
```

**검증 포인트:**
- ✅ Firestore 연결 성공
- ✅ 메모리 + Firestore 동시 저장
- ✅ 기존 기능 100% 정상 동작
- ✅ Cloud Run 재시작 시뮬레이션 (프로세스 재시작 후 상태 복구)

### **Phase 3: GCP 환경 설정**
```bash
# 1. GCP 환경 설정 (Firestore 포함)
./setup_gcp_environment.sh

# 2. 환경 변수 확인
cat .env
```

**검증 포인트:**
- ✅ Firestore API 활성화
- ✅ Firestore 데이터베이스 생성
- ✅ 서비스 계정 권한 설정
- ✅ 환경 변수 올바른 설정

### **Phase 4: Cloud Run 배포 (Firestore 비활성화)**
```bash
# 1. 환경 변수에서 Firestore 비활성화
# .env 파일에서 ENABLE_FIRESTORE_SYNC=false 설정

# 2. 배포
./deploy_to_gcp.sh

# 3. 기존 기능 테스트
python test/test_cloudrun.py
```

**검증 포인트:**
- ✅ 기존 기능 100% 정상 동작
- ✅ Firestore 코드가 실행되지 않음
- ✅ 메모리 상태 관리만 사용

### **Phase 5: Cloud Run 배포 (Firestore 활성화)**
```bash
# 1. 환경 변수에서 Firestore 활성화
# Cloud Run 환경 변수 설정: ENABLE_FIRESTORE_SYNC=true

# 2. 서비스 재배포
gcloud run services update golf-analyzer-backend \
  --set-env-vars="ENABLE_FIRESTORE_SYNC=true" \
  --region=asia-northeast3

# 3. 전체 기능 테스트
python test/test_cloudrun.py
```

**검증 포인트:**
- ✅ Firestore 연결 성공
- ✅ 상태 동기화 정상 동작
- ✅ Cloud Run 재시작 시 상태 복구
- ✅ 기존 기능 100% 정상 동작

## 🛡️ 안전장치

### **1. 환경 변수 기반 제어**
```python
# Firestore 사용 여부를 환경 변수로 완전 제어
ENABLE_FIRESTORE_SYNC=false  # 비활성화
ENABLE_FIRESTORE_SYNC=true   # 활성화
```

### **2. 실패 시 자동 폴백**
```python
# Firestore 실패해도 기존 메모리 상태는 정상 동작
try:
    sync_to_firestore(task_id, data)
except Exception:
    # 조용히 실패, 기존 기능 보호
    pass
```

### **3. 점진적 기능 활성화**
1. **1단계**: 메모리만 사용 (기존과 동일)
2. **2단계**: 메모리 + Firestore 저장 (읽기는 메모리 우선)
3. **3단계**: Firestore 복구 기능 활성화

### **4. 실시간 모니터링**
```bash
# Firestore 상태 실시간 확인
curl https://golf-analyzer-backend-xxx.run.app/api/firestore/status

# 시스템 정보에서 Firestore 상태 확인
curl https://golf-analyzer-backend-xxx.run.app/api/info
```

## 🚨 롤백 계획

### **즉시 롤백 (환경 변수)**
```bash
# Firestore만 비활성화 (코드 변경 없이)
gcloud run services update golf-analyzer-backend \
  --set-env-vars="ENABLE_FIRESTORE_SYNC=false" \
  --region=asia-northeast3
```

### **완전 롤백 (이전 버전)**
```bash
# 이전 Docker 이미지로 롤백
./deploy_to_gcp.sh previous-tag
```

## ✅ 성공 기준

1. **기존 기능 보존**: 모든 API 엔드포인트 정상 동작
2. **Firestore 통합**: 상태 동기화 및 복구 기능 동작
3. **성능 유지**: 응답 시간 기존과 동일
4. **안정성**: Cloud Run 재시작 시 상태 복구
5. **모니터링**: Firestore 상태 실시간 확인 가능

## 📞 문제 발생시 대응

1. **Firestore 연결 실패**: 환경 변수로 비활성화
2. **성능 저하**: Firestore 동기화 비활성화
3. **기존 기능 오류**: 즉시 이전 버전으로 롤백
4. **상태 동기화 오류**: 메모리 상태만 사용하도록 전환

이 계획을 통해 **기존 코드의 안정성을 100% 보장**하면서 Firestore 기능을 안전하게 추가할 수 있습니다. 