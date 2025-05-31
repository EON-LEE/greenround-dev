import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import cv2
from pose_estimation import PoseEstimator
import logging

logger = logging.getLogger(__name__)

class SwingAnalyzer:
    """골프 스윙을 분석하는 클래스"""

    def __init__(self):
        self.pose_estimator = PoseEstimator()

    def analyze_video(self, video_path: str) -> Dict:
        """비디오를 분석하고 결과를 반환합니다."""
        logger.info("Starting video analysis")
        
        # 비디오 캡처 초기화
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("비디오를 열 수 없습니다.")
        
        try:
            # 프레임 데이터 수집
            frames_data = self._collect_frames(cap)
            if not frames_data:
                raise ValueError("유효한 프레임을 찾을 수 없습니다.")
            
            # 키 프레임 감지
            key_frames = self._detect_key_frames(frames_data)
            logger.debug(f"Detected key frames: {key_frames}")
            
            # 스윙 분석 수행
            analysis_result = self._analyze_swing(frames_data, key_frames)
            logger.debug(f"Analysis result: {analysis_result}")
            
            return analysis_result
            
        finally:
            cap.release()

    def _collect_frames(self, cap) -> List[Dict]:
        """비디오에서 프레임 데이터를 수집합니다."""
        frames_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            processed_frame, landmarks = self.pose_estimator.process_frame(frame)
            if landmarks:
                # 각도 계산
                angles = self.pose_estimator.calculate_angles(landmarks)
                
                # 프레임 데이터 저장
                frame_data = {
                    'frame_number': frame_count,
                    'angles': angles,
                    'landmarks': self._landmarks_to_dict(landmarks)
                }
                frames_data.append(frame_data)
            
            frame_count += 1
            
        logger.info(f"Collected {len(frames_data)} valid frames")
        return frames_data

    def _detect_key_frames(self, frames_data: List[Dict]) -> Dict[str, int]:
        """주요 스윙 단계의 프레임을 감지합니다."""
        if not frames_data:
            return {}
            
        # 기본값 설정
        key_frames = {
            'address': None,
            'backswing': None,
            'top': None,
            'impact': None,
            'follow_through': None,
            'finish': None
        }
        
        # 어드레스 프레임 감지
        address_frame = self._detect_address_frame(frames_data)
        key_frames['address'] = address_frame if address_frame is not None else 0
        
        # 각도 변화를 기반으로 키 프레임 감지
        max_shoulder_rotation = 0
        min_arm_angle = float('inf')
        
        for i, frame in enumerate(frames_data):
            if i < key_frames['address']:  # 어드레스 이전 프레임은 무시
                continue
                
            angles = frame['angles']
            shoulder_angle = angles.get('shoulder_angle', 0)
            arm_angle = angles.get('right_arm', 0)
            
            # 백스윙 감지 (어깨 회전 증가)
            if shoulder_angle > max_shoulder_rotation:
                max_shoulder_rotation = shoulder_angle
                key_frames['backswing'] = i
            
            # 탑 감지 (팔 각도 최소)
            if arm_angle < min_arm_angle:
                min_arm_angle = arm_angle
                key_frames['top'] = i
        
        # 나머지 키 프레임 설정
        if key_frames['top'] is not None:
            remaining_frames = len(frames_data) - key_frames['top']
            key_frames['impact'] = key_frames['top'] + (remaining_frames // 3)
            key_frames['follow_through'] = key_frames['impact'] + (remaining_frames // 3)
            key_frames['finish'] = len(frames_data) - 1
        
        return key_frames

    def _detect_address_frame(self, frames_data: List[Dict]) -> Optional[int]:
        """어드레스 프레임을 감지합니다."""
        STABLE_FRAMES_REQUIRED = 5  # 안정된 자세로 판단하기 위한 연속 프레임 수
        ANGLE_THRESHOLD = 5  # 각도 변화 허용 범위 (도)
        POSITION_THRESHOLD = 0.05  # 위치 변화 허용 범위
        
        stable_frame_count = 0
        last_angles = None
        last_positions = None
        
        for i, frame in enumerate(frames_data):
            current_angles = frame['angles']
            current_positions = {
                'right_shoulder': np.array(frame['landmarks']['right_shoulder']),
                'left_shoulder': np.array(frame['landmarks']['left_shoulder']),
                'right_hip': np.array(frame['landmarks']['right_hip']),
                'left_hip': np.array(frame['landmarks']['left_hip'])
            }
            
            # 첫 프레임이면 기준값으로 설정
            if last_angles is None:
                last_angles = current_angles
                last_positions = current_positions
                continue
            
            # 자세 안정성 체크
            is_stable = True
            
            # 각도 안정성 체크
            for angle_key in ['right_arm', 'left_arm', 'spine_angle']:
                if abs(current_angles.get(angle_key, 0) - last_angles.get(angle_key, 0)) > ANGLE_THRESHOLD:
                    is_stable = False
                    break
            
            # 위치 안정성 체크
            if is_stable:
                for pos_key in current_positions:
                    if np.linalg.norm(current_positions[pos_key] - last_positions[pos_key]) > POSITION_THRESHOLD:
                        is_stable = False
                        break
            
            # 어드레스 조건 체크
            if is_stable:
                right_arm_angle = current_angles.get('right_arm', 0)
                spine_angle = current_angles.get('spine_angle', 0)
                
                # 어드레스 자세 조건
                if (160 <= right_arm_angle <= 180 and  # 팔이 거의 펴진 상태
                    20 <= spine_angle <= 45):          # 적절한 척추 각도
                    stable_frame_count += 1
                else:
                    stable_frame_count = 0
            else:
                stable_frame_count = 0
            
            # 충분한 시간 동안 안정된 자세가 유지되면 어드레스로 판단
            if stable_frame_count >= STABLE_FRAMES_REQUIRED:
                return i - STABLE_FRAMES_REQUIRED + 1
            
            last_angles = current_angles
            last_positions = current_positions
        
        return None  # 어드레스 프레임을 찾지 못한 경우

    def _analyze_swing(self, frames_data: List[Dict], key_frames: Dict[str, int]) -> Dict:
        """스윙을 분석하고 메트릭스와 평가를 생성합니다."""
        # 메트릭스 계산
        metrics = self._calculate_metrics(frames_data, key_frames)
        
        # 평가 생성
        evaluations = self._evaluate_swing(frames_data, key_frames, metrics)
        
        return {
            'frames': frames_data,
            'key_frames': key_frames,
            'metrics': metrics,
            'evaluations': evaluations
        }

    def _calculate_metrics(self, frames_data: List[Dict], key_frames: Dict[str, int]) -> Dict[str, float]:
        """스윙 메트릭스를 계산합니다."""
        logger.debug("Starting metrics calculation...")
        
        # 기본 메트릭스 초기화
        metrics = {
            'shoulder_rotation': 0.0,
            'hip_rotation': 0.0,
            'head_movement': 0.0,
            'backswing_angle': 0.0,
            'impact_angle': 0.0,
            'follow_through_angle': 0.0
        }
        
        if not frames_data:
            logger.warning("No frames data available for metrics calculation")
            return metrics
            
        logger.debug(f"Processing {len(frames_data)} frames for metrics")
        
        # 최대 회전 각도 계산
        for frame in frames_data:
            angles = frame['angles']
            metrics['shoulder_rotation'] = max(metrics['shoulder_rotation'], angles.get('shoulder_angle', 0))
            metrics['hip_rotation'] = max(metrics['hip_rotation'], angles.get('hip_angle', 0))
        
        logger.debug(f"Rotation metrics: shoulder={metrics['shoulder_rotation']}, hip={metrics['hip_rotation']}")
        
        # 헤드 무브먼트 계산
        try:
            initial_frame = frames_data[0]
            if 'landmarks' in initial_frame and 'nose' in initial_frame['landmarks'] and initial_frame['landmarks']['nose'] is not None:
                initial_head_pos = np.array(initial_frame['landmarks']['nose'])
                max_movement = 0.0
                
                for frame in frames_data[1:]:
                    if 'landmarks' in frame and 'nose' in frame['landmarks'] and frame['landmarks']['nose'] is not None:
                        current_head_pos = np.array(frame['landmarks']['nose'])
                        movement = np.linalg.norm(current_head_pos - initial_head_pos)
                        max_movement = max(max_movement, movement)
                
                metrics['head_movement'] = float(max_movement)
                logger.debug(f"Calculated head movement: {metrics['head_movement']}")
            else:
                logger.warning("Initial frame missing nose landmark, setting head_movement to 0.0")
        except Exception as e:
            logger.error(f"Error calculating head movement: {str(e)}")
            metrics['head_movement'] = 0.0
        
        # 주요 각도 계산 - None 값 처리 추가
        try:
            if key_frames.get('top') is not None and 0 <= key_frames['top'] < len(frames_data):
                metrics['backswing_angle'] = frames_data[key_frames['top']]['angles'].get('right_arm', 0)
            if key_frames.get('impact') is not None and 0 <= key_frames['impact'] < len(frames_data):
                metrics['impact_angle'] = frames_data[key_frames['impact']]['angles'].get('right_arm', 0)
            if key_frames.get('finish') is not None and 0 <= key_frames['finish'] < len(frames_data):
                metrics['follow_through_angle'] = frames_data[key_frames['finish']]['angles'].get('right_arm', 0)
        except Exception as e:
            logger.error(f"Error calculating swing angles: {str(e)}")
        
        logger.debug(f"Final calculated metrics: {metrics}")
        return metrics

    def _evaluate_swing(self, frames_data: List[Dict], key_frames: Dict[str, int], metrics: Dict[str, float]) -> Dict[str, Dict[str, bool]]:
        """스윙을 평가합니다."""
        logger.debug("Starting swing evaluation...")
        logger.debug(f"Input metrics: {metrics}")
        
        evaluations = {}
        
        # 어드레스 평가
        address_idx = key_frames.get('address')
        if address_idx is not None and 0 <= address_idx < len(frames_data):
            address_frame = frames_data[address_idx]
            evaluations['address'] = {
                'Arm Angle Straight': 165 <= address_frame['angles'].get('right_arm', 0) <= 180,
                'Posture Stable': address_frame['angles'].get('spine_angle', 0) >= 30
            }
        
        # 탑 평가
        top_idx = key_frames.get('top')
        if top_idx is not None and 0 <= top_idx < len(frames_data):
            evaluations['top'] = {
                'Shoulder Rotation Good': metrics.get('shoulder_rotation', 0) >= 80,
                'Head Stable': metrics.get('head_movement', 0) <= 0.1
            }
        
        # 임팩트 평가
        impact_idx = key_frames.get('impact')
        if impact_idx is not None and 0 <= impact_idx < len(frames_data):
            impact_frame = frames_data[impact_idx]
            evaluations['impact'] = {
                'Arm Straight': 165 <= impact_frame['angles'].get('right_arm', 0) <= 180,
                'Hip Rotation Good': metrics.get('hip_rotation', 0) >= 45
            }
        
        # 팔로우 스루 평가
        follow_through_idx = key_frames.get('follow_through')
        if follow_through_idx is not None and 0 <= follow_through_idx < len(frames_data):
            follow_frame = frames_data[follow_through_idx]
            evaluations['follow_through'] = {
                'Follow Through Complete': follow_frame['angles'].get('right_arm', 0) <= 120,
                'Balance Maintained': follow_frame['angles'].get('right_leg', 0) <= 160
            }
        
        # 피니시 평가
        finish_idx = key_frames.get('finish')
        if finish_idx is not None and 0 <= finish_idx < len(frames_data):
            finish_frame = frames_data[finish_idx]
            evaluations['finish'] = {
                'Follow Through Complete': finish_frame['angles'].get('right_arm', 0) <= 120,
                'Balance Maintained': finish_frame['angles'].get('right_leg', 0) <= 160
            }
        
        logger.debug(f"Final evaluations: {evaluations}")
        return evaluations

    def _landmarks_to_dict(self, landmarks) -> Dict[str, List[float]]:
        """랜드마크를 딕셔너리로 변환합니다."""
        return {
            'nose': landmarks.nose.tolist() if hasattr(landmarks, 'nose') else None,
            'left_shoulder': landmarks.left_shoulder.tolist(),
            'right_shoulder': landmarks.right_shoulder.tolist(),
            'left_elbow': landmarks.left_elbow.tolist(),
            'right_elbow': landmarks.right_elbow.tolist(),
            'left_wrist': landmarks.left_wrist.tolist(),
            'right_wrist': landmarks.right_wrist.tolist(),
            'left_hip': landmarks.left_hip.tolist(),
            'right_hip': landmarks.right_hip.tolist(),
            'left_knee': landmarks.left_knee.tolist(),
            'right_knee': landmarks.right_knee.tolist(),
            'left_ankle': landmarks.left_ankle.tolist(),
            'right_ankle': landmarks.right_ankle.tolist()
        } 