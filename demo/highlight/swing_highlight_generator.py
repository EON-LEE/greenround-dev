import numpy as np
from typing import Dict, List, Tuple
import logging
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.video.fx import all as vfx
import cv2


from swing_analyzer_3d import SwingAnalyzer3D
from video_processor import VideoProcessor

logger = logging.getLogger(__name__)

class SwingHighlightGenerator:
    def __init__(self):
        """스윙 하이라이트 생성기 초기화"""
        self.analyzer = SwingAnalyzer3D()
        self.video_processor = VideoProcessor()
        self.highlight_points = []
        self.highlight_segments = []
        
    def analyze_swing(self, video_path: str) -> Dict:
        """스윙 분석 및 하이라이트 포인트 식별"""
        try:
            # 비디오 처리 및 프레임 추출
            frames_data = self.video_processor.process_video(video_path)
            
            # 스윙 분석 수행
            analysis_result = self.analyzer.analyze_swing(frames_data)
            
            # 하이라이트 포인트 식별
            self._identify_highlight_points(frames_data, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in analyze_swing: {str(e)}")
            return {}
            
    def _identify_highlight_points(self, frames_data: List[Dict], analysis_result: Dict):
        """하이라이트 포인트 식별"""
        try:
            key_frames = analysis_result['key_frames']
            evaluations = analysis_result['evaluations']
            
            # 각 단계별 하이라이트 포인트 식별
            for phase, frame_idx in key_frames.items():
                if self._is_highlight_worthy(frames_data[frame_idx], evaluations[phase]):
                    self.highlight_points.append({
                        'phase': phase,
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / len(frames_data),
                        'metrics': self._extract_highlight_metrics(frames_data[frame_idx])
                    })
                    
        except Exception as e:
            logger.error(f"Error identifying highlight points: {str(e)}")
            
    def _is_highlight_worthy(self, frame_data: Dict, evaluation: Dict) -> bool:
        """프레임이 하이라이트 가치가 있는지 평가"""
        try:
            # 각 단계별 평가 기준
            criteria = {
                'address': ['Posture', 'Knee Flex', 'Setup Balance'],
                'backswing': ['Shoulder Turn', 'Hip Resistance', 'Spine Angle'],
                'top': ['Shoulder Turn', 'Arm Position', 'Wrist Hinge'],
                'impact': ['Hip Rotation', 'Spine Angle', 'Weight Transfer'],
                'follow_through': ['Body Rotation', 'Hip Clearance', 'Extension'],
                'finish': ['Balance', 'Full Rotation', 'Posture']
            }
            
            # 평가 기준의 70% 이상이 True인 경우 하이라이트로 선정
            phase = frame_data.get('phase', '')
            if phase in criteria:
                true_count = sum(1 for criterion in criteria[phase] if evaluation.get(criterion, False))
                return true_count / len(criteria[phase]) >= 0.7
                
            return False
            
        except Exception as e:
            logger.error(f"Error in highlight evaluation: {str(e)}")
            return False
            
    def _extract_highlight_metrics(self, frame_data: Dict) -> Dict:
        """하이라이트 프레임의 주요 메트릭스 추출"""
        try:
            return {
                'angles': frame_data.get('angles', {}),
                'landmarks': frame_data.get('landmarks', {}),
                'phase': frame_data.get('phase', '')
            }
        except Exception as e:
            logger.error(f"Error extracting highlight metrics: {str(e)}")
            return {}
            
    def generate_highlights(self, video_path: str, output_path: str) -> bool:
        """하이라이트 영상 생성"""
        try:
            # 비디오 클립 로드
            video_clip = VideoFileClip(video_path)
            
            # 하이라이트 세그먼트 생성
            highlight_clips = []
            for point in self.highlight_points:
                # 하이라이트 구간 전후 0.5초 포함
                start_time = max(0, point['timestamp'] - 0.5)
                end_time = min(video_clip.duration, point['timestamp'] + 0.5)
                
                # 하이라이트 클립 생성
                clip = video_clip.subclip(start_time, end_time)
                
                # 임팩트 구간은 슬로우 모션 적용
                if point['phase'] == 'impact':
                    clip = clip.fx(vfx.speedx, 0.5)
                
                # 텍스트 오버레이 추가
                text_clip = TextClip(
                    f"{point['phase'].upper()}\n" + 
                    self._format_metrics(point['metrics']),
                    fontsize=24,
                    color='white',
                    bg_color='black',
                    font='Arial'
                ).set_position(('right', 'top')).set_duration(clip.duration)
                
                # 클립 합성
                final_clip = CompositeVideoClip([clip, text_clip])
                highlight_clips.append(final_clip)
            
            # 모든 하이라이트 클립 연결
            final_video = concatenate_videoclips(highlight_clips)
            
            # 최종 영상 저장
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=30
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating highlights: {str(e)}")
            return False
            
    def _format_metrics(self, metrics: Dict) -> str:
        """메트릭스 포맷팅"""
        try:
            formatted_text = []
            angles = metrics.get('angles', {})
            
            # 주요 각도 정보 포맷팅
            for angle_name, value in angles.items():
                if isinstance(value, (int, float)):
                    formatted_text.append(f"{angle_name}: {value:.1f}°")
                    
            return "\n".join(formatted_text)
            
        except Exception as e:
            logger.error(f"Error formatting metrics: {str(e)}")
            return "" 