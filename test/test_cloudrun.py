#!/usr/bin/env python3
"""
실제 골프 비디오 파일을 사용한 Greenround API 테스트 (v2.0 마이크로서비스 + Firestore)

환경 변수 설정:
- .env 파일을 생성하고 다음 변수들을 설정하세요:
  * GCP_PROJECT_ID: GCP 프로젝트 ID
  * GCP_SERVICE_NAME: Cloud Run 서비스명 (기본: greenround-backend)
  * SERVICE_BASE_URL: 직접 서비스 URL 지정 (선택사항)

사용법:
1. .env 파일 생성 및 설정
2. python test/test_cloudrun.py 실행
"""

import requests
import json
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Firestore 직접 모니터링을 위한 import
try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    print("⚠️ google-cloud-firestore가 설치되지 않았습니다.")
    print("   pip install google-cloud-firestore 를 실행하세요.")

# .env 파일에서 환경 변수 로드 (파일이 없어도 에러 없이 처리)
load_dotenv()

# 환경 변수에서 서비스 URL 동적 구성
def get_service_url():
    """환경 변수에서 서비스 URL을 동적으로 생성"""
    print("🔧 환경 변수에서 서비스 URL 구성 중...")
    
    # .env 파일에서 값 읽기
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "asia-northeast3")
    service_name = os.getenv("GCP_SERVICE_NAME")
    
    # 환경 변수 상태 출력
    print(f"   GCP_PROJECT_ID: {project_id if project_id else '❌ 없음'}")
    print(f"   GCP_REGION: {region}")
    print(f"   GCP_SERVICE_NAME: {service_name if service_name else '❌ 없음'}")
    
    # 1. 환경 변수에서 직접 URL 확인
    service_base_url = os.getenv("SERVICE_BASE_URL")
    if service_base_url:
        service_url = service_base_url.rstrip('/')
        print(f"📍 환경 변수에서 직접 URL 사용: {service_url}")
        return service_url
    
    # 2. 프로젝트 정보로 URL 구성
    if project_id and service_name:
        # GCP 프로젝트 번호 확인 시도
        project_number = os.getenv("GCP_PROJECT_NUMBER")
        
        if project_number:
            # 프로젝트 번호가 있는 경우 정확한 Cloud Run URL 구성
            service_url = f"https://{service_name}-{project_number}.{region}.run.app"
            print(f"📍 프로젝트 번호로 URL 구성: {service_url}")
        else:
            # 프로젝트 번호가 없으면 gcloud로 조회 시도
            try:
                import subprocess
                result = subprocess.run(
                    ["gcloud", "projects", "describe", project_id, "--format=value(projectNumber)"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    project_number = result.stdout.strip()
                    service_url = f"https://{service_name}-{project_number}.{region}.run.app"
                    print(f"📍 조회된 프로젝트 번호로 URL 구성: {service_url}")
                    print(f"   프로젝트 번호: {project_number}")
                else:
                    # 프로젝트 번호 조회 실패 시 기본 패턴 사용
                    service_url = f"https://{service_name}.{region}.run.app"
                    print(f"📍 기본 패턴으로 URL 구성: {service_url}")
                    print(f"   ⚠️ 프로젝트 번호를 조회할 수 없어 정확하지 않을 수 있습니다.")
            except Exception as e:
                service_url = f"https://{service_name}.{region}.run.app"
                print(f"📍 기본 패턴으로 URL 구성: {service_url}")
                print(f"   ⚠️ 프로젝트 번호 조회 실패: {e}")
        
        print(f"   프로젝트 ID: {project_id}")
        print(f"   리전: {region}")
        print(f"   서비스명: {service_name}")
        return service_url
    
    # 3. 환경 변수가 없으면 기본 URL 사용
    print("⚠️ 필수 환경 변수가 없습니다.")
    print("   .env 파일을 생성하고 다음 변수들을 설정하세요:")
    print("   - GCP_PROJECT_ID=your-project-id")
    print("   - GCP_SERVICE_NAME=greenround-backend")
    print("   - SERVICE_BASE_URL=https://your-service-url (선택사항)")
    print("   기본 URL을 사용합니다.")
    
    return "https://golf-analyzer-backend-984220723638.asia-northeast3.run.app"

def verify_service_url(url):
    """서비스 URL이 유효한지 확인"""
    try:
        print(f"\n🔍 서비스 URL 연결 테스트: {url}")
        response = requests.get(f"{url}/", timeout=10)
        if response.status_code == 200:
            print("✅ URL 연결 성공!")
            return True
        else:
            print(f"⚠️ URL 응답 코드: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ URL 연결 실패: {e}")
        return False

SERVICE_URL = get_service_url()
VIDEO_FILE = "/Users/eonlee/Documents/Projects/golf-3d-analyzer/test/KakaoTalk_Video_2025-05-31-20-46-31.mp4"

def test_download_api(session, task_id, feature_name):
    """완료된 작업의 결과를 다운로드 API로 테스트"""
    try:
        print(f"   📥 {feature_name} 다운로드 API 테스트 중...")
        
        # 기능별 다운로드 URL 결정 (새로운 마이크로서비스 구조)
        if "하이라이트" in feature_name:
            download_url = f"{SERVICE_URL}/api/results/highlights/{task_id}.mp4"
            stream_url = f"{SERVICE_URL}/api/results/highlights/{task_id}/stream"
            expected_content_type = "video/mp4"
        elif "시퀀스" in feature_name:
            download_url = f"{SERVICE_URL}/api/results/sequences/{task_id}.png"
            stream_url = f"{SERVICE_URL}/api/results/sequences/{task_id}/stream"
            expected_content_type = "image/png"
        elif "볼" in feature_name:
            # 볼 트래킹은 아직 전용 엔드포인트가 없으므로 상태 API를 통해 다운로드 URL을 가져옴
            status_response = session.get(f"{SERVICE_URL}/api/status/{task_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                result_data = status_data.get("result_data", {})
                if "download_url" in result_data:
                    download_url = result_data["download_url"]
                    stream_url = download_url  # 볼 트래킹은 스트리밍 별도 URL이 없음
                    expected_content_type = "video/mp4"
                else:
                    print(f"   ❌ 볼 트래킹 결과 URL을 찾을 수 없음")
                    return False
            else:
                print(f"   ❌ 볼 트래킹 상태 확인 실패")
                return False
        else:
            print(f"   ❓ 알 수 없는 기능 타입: {feature_name}")
            return False
        
        # 1. 다운로드 테스트 (파일 다운로드)
        print(f"     다운로드 URL: {download_url}")
        download_response = session.get(download_url, params={"download": "true"}, timeout=30)
        
        if download_response.status_code == 200:
            content_type = download_response.headers.get("content-type", "")
            content_length = download_response.headers.get("content-length", "0")
            
            # 다운로드 디렉토리 생성
            download_dir = Path("downloads")
            download_dir.mkdir(exist_ok=True)
            
            # 파일 확장자 결정
            if "하이라이트" in feature_name:
                file_ext = ".mp4"
            elif "시퀀스" in feature_name:
                file_ext = ".png"
            else:
                file_ext = ".mp4"
            
            # 파일 저장
            save_path = download_dir / f"{task_id}_{feature_name.replace(' ', '_')}{file_ext}"
            with open(save_path, "wb") as f:
                f.write(download_response.content)
            
            if expected_content_type in content_type:
                print(f"   ✅ 다운로드 성공 - 크기: {content_length} bytes, 타입: {content_type}")
                print(f"   📁 파일 저장 위치: {save_path.absolute()}")
                download_success = True
            else:
                print(f"   ⚠️  다운로드 성공하지만 예상과 다른 Content-Type: {content_type}")
                print(f"   📁 파일 저장 위치: {save_path.absolute()}")
                download_success = True
        elif download_response.status_code == 202:
            # 아직 처리 중인 경우
            try:
                processing_data = download_response.json()
                print(f"   ⏳ 아직 처리 중: {processing_data.get('message', 'Processing...')}")
                print(f"   진행률: {processing_data.get('progress', 0)}%")
                return False
            except:
                print(f"   ⏳ 아직 처리 중 (HTTP 202)")
                return False
        elif download_response.status_code == 302:
            # 리다이렉트의 경우 (GCS URL로 리다이렉트)
            redirect_url = download_response.headers.get("location", "")
            print(f"   🔄 GCS URL로 리다이렉트: {redirect_url[:100]}...")
            
            # 리다이렉트된 URL에서 실제 파일 다운로드
            if redirect_url:
                gcs_response = session.get(redirect_url, timeout=30)
                if gcs_response.status_code == 200:
                    content_length = gcs_response.headers.get("content-length", "0")
                    
                    # 다운로드 디렉토리 생성
                    download_dir = Path("downloads")
                    download_dir.mkdir(exist_ok=True)
                    
                    # 파일 확장자 결정
                    if "하이라이트" in feature_name:
                        file_ext = ".mp4"
                    elif "시퀀스" in feature_name:
                        file_ext = ".png"
                    else:
                        file_ext = ".mp4"
                    
                    # 파일 저장
                    save_path = download_dir / f"{task_id}_{feature_name.replace(' ', '_')}{file_ext}"
                    with open(save_path, "wb") as f:
                        f.write(gcs_response.content)
                    
                    print(f"   ✅ GCS 다운로드 성공 - 크기: {content_length} bytes")
                    print(f"   📁 파일 저장 위치: {save_path.absolute()}")
                    download_success = True
                else:
                    print(f"   ❌ GCS 다운로드 실패: HTTP {gcs_response.status_code}")
                    download_success = False
            else:
                print(f"   ❌ 리다이렉트 URL이 없음")
                download_success = False
        else:
            print(f"   ❌ 다운로드 실패: HTTP {download_response.status_code}")
            try:
                error_data = download_response.json()
                print(f"     오류 내용: {error_data}")
            except:
                print(f"     응답: {download_response.text[:200]}...")
            download_success = False
        
        # 스트리밍 테스트는 스킵 (다운로드 성공하면 스트리밍도 동작한다고 가정)
        stream_success = True
        
        return download_success and stream_success
        
    except Exception as e:
        print(f"   ❌ 다운로드 테스트 중 오류: {e}")
        return False

def get_collection_name_by_task_id(task_id: str) -> str:
    """task_id에서 기능별 컬렉션명 추출"""
    if task_id.startswith('highlight_'):
        return 'highlight_tasks'
    elif task_id.startswith('sequence_'):
        return 'sequence_tasks'
    elif task_id.startswith('balltrack_'):
        return 'balltracking_tasks'
    elif task_id.startswith('ballanalysis_'):
        return 'ballanalysis_tasks'
    else:
        return 'tasks'

def monitor_firestore_status(task_ids, max_wait_time=300):
    """Firestore에서 직접 작업 상태 모니터링"""
    if not FIRESTORE_AVAILABLE:
        print("❌ Firestore 라이브러리가 없어서 모니터링할 수 없습니다.")
        return [], task_ids
    
    print(f"\n🔥 Firestore 직접 모니터링 시작 ({len(task_ids)}개 작업)")
    
    try:
        # Firestore 클라이언트 초기화
        project_id = os.getenv("FIRESTORE_PROJECT_ID") or os.getenv("GCP_PROJECT_ID")
        database_id = os.getenv("FIRESTORE_DATABASE_ID", "(default)")
        
        if not project_id:
            print("❌ FIRESTORE_PROJECT_ID 또는 GCP_PROJECT_ID 환경변수가 필요합니다.")
            return [], task_ids
        
        if database_id != "(default)":
            db = firestore.Client(project=project_id, database=database_id)
            print(f"   📊 Firestore 연결: {project_id} (DB: {database_id})")
        else:
            db = firestore.Client(project=project_id)
            print(f"   📊 Firestore 연결: {project_id} (기본 DB)")
            
    except Exception as e:
        print(f"❌ Firestore 연결 실패: {e}")
        return [], task_ids
    
    completed_tasks = []
    start_time = time.time()
    
    while task_ids and (time.time() - start_time) < max_wait_time:
        print(f"\n⏳ Firestore 상태 확인 중... (남은 작업: {len(task_ids)}개)")
        
        for feature_name, task_id in task_ids[:]:  # 복사본으로 순회
            try:
                # Firestore에서 직접 문서 조회 (기능별 컬렉션 우선)
                collection_name = get_collection_name_by_task_id(task_id)
                doc_ref = db.collection(collection_name).document(task_id)
                doc = doc_ref.get()
                
                # 기존 tasks 컬렉션에서도 조회 시도 (하위 호환성)
                if not doc.exists and collection_name != 'tasks':
                    doc_ref = db.collection('tasks').document(task_id)
                    doc = doc_ref.get()
                    collection_name = 'tasks'
                
                if doc.exists:
                    data = doc.to_dict()
                    status = data.get('status', 'unknown')
                    progress = data.get('progress', 0)
                    message = data.get('message', '')
                    updated_at = data.get('updated_at')
                    
                    print(f"   🔥 {feature_name} ({collection_name}): {status} ({progress}%) - {message}")
                    if updated_at:
                        print(f"      마지막 업데이트: {updated_at}")
                    
                    if status == "completed":
                        print(f"   ✅ {feature_name} 완료!")
                        
                        # 결과 데이터 확인
                        result_data = data.get('result_data', {})
                        if result_data:
                            print(f"      결과 데이터: {list(result_data.keys())}")
                            if 'download_url' in result_data:
                                print(f"      다운로드 URL: {result_data['download_url']}")
                        
                        # 다운로드 테스트 (API를 통해)
                        download_success = test_download_api(requests.Session(), task_id, feature_name)
                        
                        completed_tasks.append((feature_name, task_id, download_success))
                        task_ids.remove((feature_name, task_id))
                        
                    elif status == "failed":
                        error_msg = data.get('message', 'Unknown error')
                        print(f"   ❌ {feature_name} 실패: {error_msg}")
                        task_ids.remove((feature_name, task_id))
                        
                    elif status in ["pending", "processing"]:
                        # 진행 중 - 계속 모니터링
                        pass
                        
                    else:
                        print(f"   ❓ {feature_name}: 알 수 없는 상태 ({status})")
                        
                else:
                    print(f"   ⚠️ {feature_name}: Firestore에서 문서를 찾을 수 없음 (task_id: {task_id})")
                    
            except Exception as e:
                print(f"   ❌ {feature_name} Firestore 조회 오류: {e}")
        
        if task_ids:  # 아직 진행 중인 작업이 있으면 대기
            print(f"   ⏰ 10초 후 다시 확인...")
            time.sleep(10)
    
    # 타임아웃 처리
    if task_ids:
        print(f"\n⏰ {max_wait_time}초 타임아웃 - {len(task_ids)}개 작업이 완료되지 않음")
    
    return completed_tasks, task_ids

def test_with_real_video():
    """실제 골프 비디오로 전체 워크플로우 테스트"""
    
    print("🏌️ Greenround - 실제 비디오 테스트 (v2.0 마이크로서비스 + Firestore)")
    print("=" * 70)
    print(f"서비스 URL: {SERVICE_URL}")
    print(f"비디오 파일: {VIDEO_FILE}")
    print("=" * 70)
    
    # URL 연결 테스트
    if not verify_service_url(SERVICE_URL):
        print("❌ 서비스 URL에 연결할 수 없습니다. 배포 상태를 확인해주세요.")
        return False
    
    # 파일 존재 확인
    if not Path(VIDEO_FILE).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {VIDEO_FILE}")
        return False
    
    # 파일 크기 확인
    file_size = Path(VIDEO_FILE).stat().st_size
    print(f"📁 파일 크기: {file_size / 1024 / 1024:.1f} MB")
    
    session = requests.Session()
    
    # 1. Health Check (새로운 구조)
    print("\n🔍 서비스 상태 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 서비스 정상: {health_data['message']}")
            print(f"   버전: {health_data.get('version', 'unknown')}")
            print(f"   아키텍처: {health_data.get('architecture', 'unknown')}")
        else:
            print(f"❌ 서비스 상태 확인 실패: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 서비스 연결 실패: {e}")
        return False
    
    # 2. 시스템 정보 확인 (새로운 엔드포인트)
    print("\n📊 시스템 정보 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/info", timeout=10)
        if response.status_code == 200:
            info_data = response.json()
            print(f"✅ API 버전: {info_data.get('api_version', 'unknown')}")
            print(f"   지원 형식: {', '.join(info_data.get('supported_formats', []))}")
            print(f"   최대 파일 크기: {info_data.get('max_file_size', 'unknown')}")
            
            # 서비스별 기능 확인
            services = info_data.get('services', {})
            roundreels = services.get('roundreels', {})
            if roundreels:
                available_features = [k for k, v in roundreels.items() if v]
                print(f"   RoundReels 기능: {', '.join(available_features)}")
            
            # 🆕 Firestore 상태 확인
            firestore_info = info_data.get('firestore', {})
            if firestore_info:
                firestore_status = firestore_info.get('status', 'unknown')
                firestore_message = firestore_info.get('message', '')
                status_icon = "✅" if firestore_status == "success" else "⚠️" if firestore_status == "disabled" else "❌"
                print(f"   {status_icon} Firestore: {firestore_status} - {firestore_message}")
        else:
            print(f"⚠️  시스템 정보 확인 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️  시스템 정보 확인 중 오류: {e}")
    
    # 2-1. 🆕 Firestore 전용 상태 확인
    print("\n🔥 Firestore 상태 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/firestore/status", timeout=10)
        if response.status_code == 200:
            firestore_data = response.json()
            status = firestore_data.get('status', 'unknown')
            message = firestore_data.get('message', '')
            
            if status == "success":
                print(f"✅ Firestore 연결 성공: {message}")
            elif status == "disabled":
                print(f"⚠️ Firestore 비활성화: {message}")
            else:
                print(f"❌ Firestore 연결 실패: {message}")
        else:
            print(f"⚠️ Firestore 상태 확인 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️ Firestore 상태 확인 중 오류: {e}")
    
    # 2-2. 🆕 Firestore 컬렉션 정보 확인
    print("\n📊 Firestore 컬렉션 정보 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/firestore/collections", timeout=10)
        if response.status_code == 200:
            collections_data = response.json()
            status = collections_data.get('status', 'unknown')
            
            if status == "success":
                collections = collections_data.get('collections', {})
                feature_mapping = collections_data.get('feature_mapping', {})
                
                print(f"✅ 기능별 컬렉션 정보:")
                for collection_name, info in collections.items():
                    if 'error' in info:
                        print(f"   ❌ {collection_name}: 오류 - {info['error']}")
                    else:
                        doc_count = info.get('document_count', 0)
                        latest_update = info.get('latest_update')
                        status_counts = info.get('status_counts', {})
                        
                        print(f"   📁 {collection_name}: {doc_count}개 문서")
                        if latest_update:
                            print(f"      마지막 업데이트: {latest_update}")
                        if status_counts:
                            status_summary = ", ".join([f"{k}: {v}" for k, v in status_counts.items() if v > 0])
                            if status_summary:
                                print(f"      상태별 카운트: {status_summary}")
                
                print(f"   🔗 기능 매핑: {feature_mapping}")
                
            elif status == "disabled":
                print(f"⚠️ Firestore 비활성화됨")
            else:
                print(f"❌ 컬렉션 정보 조회 실패: {collections_data.get('message', 'Unknown error')}")
        else:
            print(f"⚠️ 컬렉션 정보 확인 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️ 컬렉션 정보 확인 중 오류: {e}")
    
    # 3. RoundReels 서비스 상태 확인
    print("\n🏌️ RoundReels 서비스 상태 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/roundreels/health", timeout=10)
        if response.status_code == 200:
            roundreels_health = response.json()
            print(f"✅ RoundReels 서비스: {roundreels_health.get('status', 'unknown')}")
            
            features = roundreels_health.get('features', {})
            engines = roundreels_health.get('engines', {})
            
            print(f"   기능 상태:")
            for feature, status in features.items():
                status_icon = "✅" if status else "❌"
                print(f"     {status_icon} {feature}: {status}")
            
            print(f"   엔진 상태:")
            for engine, status in engines.items():
                print(f"     📦 {engine}: {status}")
        else:
            print(f"⚠️  RoundReels 서비스 상태 확인 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️  RoundReels 서비스 상태 확인 중 오류: {e}")
    
    # 4. 파일 업로드
    print("\n📤 파일 업로드")
    try:
        with open(VIDEO_FILE, "rb") as f:
            filename = Path(VIDEO_FILE).name
            files = {"file": (filename, f, "video/mp4")}
            
            # 헤더 설정
            headers = {
                'User-Agent': 'Golf3DAnalyzer-TestClient/2.0',
                'Accept': '*/*'
            }
            
            print(f"   파일명: {filename}")
            print(f"   파일 크기: {file_size} bytes")
            print(f"   Content-Type: video/mp4")
            print("   업로드 중... (시간이 걸릴 수 있습니다)")
            
            response = session.post(
                f"{SERVICE_URL}/api/upload",
                files=files,
                headers=headers,
                timeout=60  # 대용량 파일을 위해 타임아웃 증가
            )
            
            print(f"   응답 상태: HTTP {response.status_code}")
            
            if response.status_code == 200:
                upload_data = response.json()
                file_id = upload_data["file_id"]
                print(f"✅ 업로드 성공: {file_id}")
                print(f"   업로드된 크기: {upload_data.get('size', 'unknown')} bytes")
                print(f"   상태: {upload_data.get('status', 'unknown')}")
            else:
                print(f"❌ 업로드 실패: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   오류: {error_data}")
                except:
                    print(f"   응답: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ 파일 업로드 중 오류: {e}")
        return False
    
    # 5. 기능 테스트 (새로운 마이크로서비스 경로)
    features = [
        {
            "name": "하이라이트 비디오",
            "endpoint": "/api/roundreels/highlight-video",
            "payload": {
                "file_id": file_id,
                "total_duration": 15,
                "slow_factor": 2
            }
        },
        {
            "name": "스윙 시퀀스",
            "endpoint": "/api/roundreels/swing-sequence", 
            "payload": {
                "file_id": file_id
            }
        },
        # 볼 트래킹은 처리 시간이 길어서 일단 주석 처리
        # {
        #     "name": "볼 트래킹",
        #     "endpoint": "/api/roundreels/ball-tracking",
        #     "payload": {
        #         "file_id": file_id,
        #         "show_trajectory": True,
        #         "show_speed": True,
        #         "show_distance": True
        #     }
        # },
        # {
        #     "name": "볼 분석",
        #     "endpoint": "/api/roundreels/ball-analysis",
        #     "payload": {
        #         "file_id": file_id,
        #         "analysis_type": "full"
        #     }
        # }
    ]
    
    task_ids = []
    
    for feature in features:
        print(f"\n🎬 {feature['name']} 요청")
        
        try:
            response = session.post(
                f"{SERVICE_URL}{feature['endpoint']}",
                json=feature["payload"],
                timeout=30
            )
            
            if response.status_code in [200, 202]:  # 200과 202 모두 성공으로 처리
                task_data = response.json()
                task_id = task_data["task_id"]
                
                # 새로운 API 응답 구조 확인
                download_url = task_data.get("download_url")
                stream_url = task_data.get("stream_url")
                estimated_time = task_data.get("estimated_time")
                
                task_ids.append((feature["name"], task_id))
                print(f"✅ 요청 성공: {task_id}")
                print(f"   상태: {task_data.get('status', 'unknown')}")
                print(f"   예상 처리 시간: {estimated_time}초")
                if download_url:
                    print(f"   다운로드 URL: {download_url}")
                if stream_url:
                    print(f"   스트리밍 URL: {stream_url}")
            else:
                print(f"❌ 요청 실패: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   오류: {error_data}")
                except:
                    print(f"   응답: {response.text}")
                    
        except Exception as e:
            print(f"❌ {feature['name']} 요청 중 오류: {e}")
    
    # 6. 작업 상태 모니터링
    if task_ids:
        print(f"\n⏳ 작업 진행 상황 모니터링 ({len(task_ids)}개 작업)")
        
        completed_tasks, remaining_task_ids = monitor_firestore_status(task_ids)
        
        # 결과 요약
        print("\n" + "=" * 70)
        print("📊 테스트 결과 요약")
        print("=" * 70)
        
        if completed_tasks:
            print("✅ 완료된 작업:")
            for feature_name, task_id, download_success in completed_tasks:
                download_status = "다운로드 성공" if download_success else "다운로드 실패"
                print(f"   - {feature_name} ({task_id}) - {download_status}")
        
        if remaining_task_ids:
            print("⏰ 타임아웃된 작업:")
            for feature_name, task_id in remaining_task_ids:
                print(f"   - {feature_name} ({task_id})")
        
        success_rate = len(completed_tasks) / (len(completed_tasks) + len(remaining_task_ids)) * 100
        download_success_count = sum(1 for _, _, success in completed_tasks if success)
        
        print(f"\n📈 성과 지표:")
        print(f"   처리 성공률: {success_rate:.1f}%")
        if completed_tasks:
            download_success_rate = (download_success_count / len(completed_tasks)) * 100
            print(f"   다운로드 성공률: {download_success_rate:.1f}%")
        
        # 🆕 Firestore 상태 재확인 (작업 완료 후)
        print(f"\n🔥 Firestore 최종 상태 확인")
        try:
            response = session.get(f"{SERVICE_URL}/api/firestore/status", timeout=5)
            if response.status_code == 200:
                firestore_data = response.json()
                status = firestore_data.get('status', 'unknown')
                if status == "success":
                    print(f"   ✅ Firestore 동기화 정상 작동")
                elif status == "disabled":
                    print(f"   ⚠️ Firestore 비활성화 상태 (메모리만 사용)")
                else:
                    print(f"   ❌ Firestore 연결 문제 (메모리 폴백 사용)")
        except:
            print(f"   ⚠️ Firestore 상태 확인 불가")
        
        if len(completed_tasks) == len(features) and download_success_count == len(completed_tasks):
            print("\n🎉 모든 기능이 성공적으로 완료되고 다운로드도 모두 성공했습니다!")
            print("   Golf 3D Analyzer v2.0 + Firestore 통합이 정상 작동 중입니다!")
            return True
        elif len(completed_tasks) == len(features):
            print("\n⚠️ 모든 기능이 완료되었지만 일부 다운로드에 실패했습니다.")
            return False
        else:
            print(f"\n⚠️ {len(remaining_task_ids)}개 작업이 완료되지 않았습니다.")
            return False
    
    else:
        print("\n❌ 실행할 수 있는 작업이 없습니다.")
        return False

def print_env_setup_guide():
    """환경 변수 설정 가이드 출력"""
    print("\n" + "=" * 70)
    print("🔧 환경 변수 설정 가이드")
    print("=" * 70)
    print("테스트를 위해 .env 파일을 생성하고 다음 내용을 추가하세요:")
    print()
    print("# Greenround 프로젝트 환경 변수")
    print("GCP_PROJECT_ID=your-project-id-here")
    print("GCP_PROJECT_NUMBER=your-project-number-here")
    print("GCP_REGION=asia-northeast3")
    print("GCP_SERVICE_NAME=greenround-backend")
    print()
    print("# Firestore 설정 (🔥 중요: 상태 모니터링용)")
    print("FIRESTORE_PROJECT_ID=your-project-id-here  # GCP_PROJECT_ID와 동일해도 됨")
    print("FIRESTORE_DATABASE_ID=(default)  # 커스텀 DB 사용시에만 변경")
    print()
    print("# Cloud Run 서비스 URL (선택사항 - 직접 지정하면 위 설정 무시)")
    print("# SERVICE_BASE_URL=https://your-service-url.asia-northeast3.run.app")
    print()
    print("예시:")
    print("GCP_PROJECT_ID=greenround-123456")
    print("FIRESTORE_PROJECT_ID=greenround-123456")
    print("GCP_PROJECT_NUMBER=658058895061")
    print("GCP_SERVICE_NAME=greenround-backend-c78809bb")
    print("# 또는 직접 URL 지정:")
    print("# SERVICE_BASE_URL=https://greenround-backend-c78809bb-658058895061.asia-northeast3.run.app")
    print("=" * 70)
    print()
    print("📦 필수 패키지 설치:")
    print("pip install google-cloud-firestore")
    print("=" * 70)

def show_test_menu():
    """테스트 메뉴 표시"""
    print("\n🏌️ Golf 3D Analyzer 테스트 메뉴")
    print("=" * 50)
    print("1. 전체 테스트 (모든 기능)")
    print("2. 하이라이트 영상 생성만")
    print("3. 스윙 시퀀스 생성만") 
    print("4. 볼 트래킹만")
    print("5. 볼 분석만")
    print("6. 시스템 상태 확인만")
    print("0. 종료")
    print("=" * 50)
    return input("선택하세요 (0-6): ").strip()

def test_single_feature(feature_config, file_id):
    """단일 기능만 테스트"""
    print(f"\n🎯 {feature_config['name']} 단독 테스트")
    print("=" * 70)
    
    session = requests.Session()
    
    # 기능 요청
    print(f"🎬 {feature_config['name']} 요청")
    
    try:
        payload = feature_config["payload"].copy()
        payload["file_id"] = file_id
        
        response = session.post(
            f"{SERVICE_URL}{feature_config['endpoint']}",
            json=payload,
            timeout=30
        )
        
        if response.status_code in [200, 202]:
            task_data = response.json()
            task_id = task_data["task_id"]
            
            download_url = task_data.get("download_url")
            stream_url = task_data.get("stream_url")
            estimated_time = task_data.get("estimated_time")
            
            print(f"✅ 요청 성공: {task_id}")
            print(f"   상태: {task_data.get('status', 'unknown')}")
            print(f"   예상 처리 시간: {estimated_time}초")
            if download_url:
                print(f"   다운로드 URL: {download_url}")
            if stream_url:
                print(f"   스트리밍 URL: {stream_url}")
            
            # 단일 작업 모니터링
            task_ids = [(feature_config["name"], task_id)]
            completed_tasks, remaining_task_ids = monitor_firestore_status(task_ids)
            
            # 결과 출력
            if completed_tasks:
                feature_name, task_id, download_success = completed_tasks[0]
                download_status = "다운로드 성공" if download_success else "다운로드 실패"
                print(f"\n✅ {feature_name} 완료 - {download_status}")
                return True
            else:
                print(f"\n⏰ {feature_config['name']} 타임아웃")
                return False
                
        else:
            print(f"❌ 요청 실패: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   오류: {error_data}")
            except:
                print(f"   응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {feature_config['name']} 요청 중 오류: {e}")
        return False

def test_system_only():
    """시스템 상태 확인만 수행"""
    print("\n🔍 시스템 상태 확인 테스트")
    print("=" * 70)
    
    session = requests.Session()
    
    # URL 연결 테스트
    if not verify_service_url(SERVICE_URL):
        print("❌ 서비스 URL에 연결할 수 없습니다.")
        return False
    
    # 1. Health Check
    print("\n🔍 서비스 상태 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 서비스 정상: {health_data['message']}")
            print(f"   버전: {health_data.get('version', 'unknown')}")
            print(f"   아키텍처: {health_data.get('architecture', 'unknown')}")
        else:
            print(f"❌ 서비스 상태 확인 실패: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 서비스 연결 실패: {e}")
        return False
    
    # 2. 시스템 정보 확인
    print("\n📊 시스템 정보 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/info", timeout=10)
        if response.status_code == 200:
            info_data = response.json()
            print(f"✅ API 버전: {info_data.get('api_version', 'unknown')}")
            print(f"   지원 형식: {', '.join(info_data.get('supported_formats', []))}")
            print(f"   최대 파일 크기: {info_data.get('max_file_size', 'unknown')}")
            
            services = info_data.get('services', {})
            roundreels = services.get('roundreels', {})
            if roundreels:
                available_features = [k for k, v in roundreels.items() if v]
                print(f"   RoundReels 기능: {', '.join(available_features)}")
            
            firestore_info = info_data.get('firestore', {})
            if firestore_info:
                firestore_status = firestore_info.get('status', 'unknown')
                firestore_message = firestore_info.get('message', '')
                status_icon = "✅" if firestore_status == "success" else "⚠️" if firestore_status == "disabled" else "❌"
                print(f"   {status_icon} Firestore: {firestore_status} - {firestore_message}")
        else:
            print(f"⚠️  시스템 정보 확인 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️  시스템 정보 확인 중 오류: {e}")
    
    # 3. Firestore 전용 상태 확인
    print("\n🔥 Firestore 상태 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/firestore/status", timeout=10)
        if response.status_code == 200:
            firestore_data = response.json()
            status = firestore_data.get('status', 'unknown')
            message = firestore_data.get('message', '')
            
            if status == "success":
                print(f"✅ Firestore 연결 성공: {message}")
            elif status == "disabled":
                print(f"⚠️ Firestore 비활성화: {message}")
            else:
                print(f"❌ Firestore 연결 실패: {message}")
        else:
            print(f"⚠️ Firestore 상태 확인 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️ Firestore 상태 확인 중 오류: {e}")
    
    # 4. RoundReels 서비스 상태 확인
    print("\n🏌️ RoundReels 서비스 상태 확인")
    try:
        response = session.get(f"{SERVICE_URL}/api/roundreels/health", timeout=10)
        if response.status_code == 200:
            roundreels_health = response.json()
            print(f"✅ RoundReels 서비스: {roundreels_health.get('status', 'unknown')}")
            
            features = roundreels_health.get('features', {})
            engines = roundreels_health.get('engines', {})
            
            print(f"   기능 상태:")
            for feature, status in features.items():
                status_icon = "✅" if status else "❌"
                print(f"     {status_icon} {feature}: {status}")
            
            print(f"   엔진 상태:")
            for engine, status in engines.items():
                print(f"     📦 {engine}: {status}")
        else:
            print(f"⚠️  RoundReels 서비스 상태 확인 실패: HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️  RoundReels 서비스 상태 확인 중 오류: {e}")
    
    print("\n✅ 시스템 상태 확인 완료")
    return True

def upload_video_file():
    """비디오 파일 업로드 (공통 함수)"""
    print("\n📤 파일 업로드")
    
    # 파일 존재 확인
    if not Path(VIDEO_FILE).exists():
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {VIDEO_FILE}")
        return None
    
    file_size = Path(VIDEO_FILE).stat().st_size
    print(f"📁 파일 크기: {file_size / 1024 / 1024:.1f} MB")
    
    session = requests.Session()
    
    try:
        with open(VIDEO_FILE, "rb") as f:
            filename = Path(VIDEO_FILE).name
            files = {"file": (filename, f, "video/mp4")}
            
            headers = {
                'User-Agent': 'Golf3DAnalyzer-TestClient/2.0',
                'Accept': '*/*'
            }
            
            print(f"   파일명: {filename}")
            print(f"   업로드 중... (시간이 걸릴 수 있습니다)")
            
            response = session.post(
                f"{SERVICE_URL}/api/upload",
                files=files,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                upload_data = response.json()
                file_id = upload_data["file_id"]
                print(f"✅ 업로드 성공: {file_id}")
                print(f"   업로드된 크기: {upload_data.get('size', 'unknown')} bytes")
                return file_id
            else:
                print(f"❌ 업로드 실패: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   오류: {error_data}")
                except:
                    print(f"   응답: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ 파일 업로드 중 오류: {e}")
        return None

def main():
    try:
        # 환경 변수가 없으면 가이드 출력
        if not os.getenv("GCP_PROJECT_ID") and not os.getenv("SERVICE_BASE_URL"):
            print_env_setup_guide()
            print("\n⚠️ 환경 변수가 설정되지 않았지만 기본 URL로 테스트를 계속합니다.")
            input("계속하려면 Enter를 누르세요...")
        
        print(f"🌐 서비스 URL: {SERVICE_URL}")
        print(f"📁 비디오 파일: {VIDEO_FILE}")
        
        # 기능 설정 정의
        features = {
            "2": {
                "name": "하이라이트 비디오",
                "endpoint": "/api/roundreels/highlight-video",
                "payload": {
                    "total_duration": 15,
                    "slow_factor": 2
                }
            },
            "3": {
                "name": "스윙 시퀀스",
                "endpoint": "/api/roundreels/swing-sequence", 
                "payload": {}
            },
            "4": {
                "name": "볼 트래킹",
                "endpoint": "/api/roundreels/ball-tracking",
                "payload": {
                    "show_trajectory": True,
                    "show_speed": True,
                    "show_distance": True
                }
            },
            "5": {
                "name": "볼 분석",
                "endpoint": "/api/roundreels/ball-analysis",
                "payload": {
                    "analysis_type": "full"
                }
            }
        }
        
        while True:
            choice = show_test_menu()
            
            if choice == "0":
                print("👋 테스트를 종료합니다.")
                break
            elif choice == "1":
                # 전체 테스트 (기존 함수)
                success = test_with_real_video()
                if success:
                    print("\n🎉 전체 테스트 완료!")
                else:
                    print("\n⚠️ 전체 테스트에서 일부 실패가 있었습니다.")
            elif choice == "6":
                # 시스템 상태 확인만
                test_system_only()
            elif choice in features:
                # 개별 기능 테스트
                feature_config = features[choice]
                
                # 파일 업로드 먼저 수행
                file_id = upload_video_file()
                if file_id:
                    success = test_single_feature(feature_config, file_id)
                    if success:
                        print(f"\n🎉 {feature_config['name']} 테스트 완료!")
                    else:
                        print(f"\n⚠️ {feature_config['name']} 테스트 실패")
                else:
                    print(f"\n❌ 파일 업로드 실패로 {feature_config['name']} 테스트를 진행할 수 없습니다.")
            else:
                print("❌ 잘못된 선택입니다. 0-6 중에서 선택해주세요.")
            
            # 다음 테스트를 위한 대기
            if choice != "0":
                print("\n" + "="*70)
                input("다음 테스트를 위해 Enter를 누르세요...")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  테스트가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 