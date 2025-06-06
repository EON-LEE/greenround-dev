# # app.py

# # --- 필요한 라이브러리 설치 ---
# # 터미널에서 한 번만 실행하세요:
# # pip install streamlit opencv-python mediapipe moviepy numpy

# import streamlit as st
# import cv2
# import numpy as np
# from pathlib import Path

# # MediaPipe Pose: 손목 좌표 추출용
# try:
#     import mediapipe as mp
# except ImportError:
#     mp = None

# # MoviePy: 슬로우 모션 처리용
# try:
#     from moviepy.editor import VideoFileClip, vfx, concatenate_videoclips
# except ImportError:
#     VideoFileClip = None

# st.set_page_config(page_title="골프 스윙 궤적 트래킹 (비학습)", layout="wide")
# st.title("골프 스윙 슬로우 모션 + 공·클럽 헤드 트래킹 (학습 불필요)")
# st.write(
#     """
#     1) 골프 스윙 영상을 업로드하면,  
#     2) 스윙 구간을 자동 탐지해 슬로우 모션 처리하고,  
#     3) 슬로우 모션 구간 동안 **골프공(흰색 원)과 클럽 헤드(빠르게 움직이는 포인트)**를  
#        HSV 마스크 + HoughCircles, 그리고 손목 인근 Optical Flow로 추적하여 부드러운 궤적을 표시합니다.  
#     4) 결과 영상을 즉시 재생하고 다운로드할 수 있습니다.  
#     """
# )

# uploaded_file = st.file_uploader("골프 스윙 영상 업로드 (MP4)", type=["mp4"])
# mode = st.radio(
#     "슬로우 모션 처리 방식",
#     options=["Local (On-device) - MoviePy 우선 사용", "Local (OpenCV 프레임 중복)"],
#     index=0,
#     help="MoviePy가 설치되어 있으면 첫 번째 옵션으로 깔끔한 슬로우 모션을 적용합니다."
# )

# if uploaded_file is not None:
#     # (1) 업로드된 영상 임시 저장
#     video_path = "input_video.mp4"
#     with open(video_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("영상 업로드 완료! 잠시만 기다려주세요...")

#     # (2) 모든 프레임 읽기
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     frames = []
#     success, frame = cap.read()
#     while success:
#         frames.append(frame)
#         success, frame = cap.read()
#     cap.release()

#     if total_frames == 0 or not frames:
#         st.error("비디오 읽기 실패. 유효한 MP4 파일인지 확인해주세요.")
#         st.stop()

#     # (3) 스윙 구간 자동 탐지 (프레임 간 픽셀 차이 기반)
#     motion = np.zeros(len(frames))
#     prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
#     for i in range(1, len(frames)):
#         gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
#         diff = cv2.absdiff(gray, prev_gray)
#         motion[i] = np.sum(diff)
#         prev_gray = gray

#     motion = np.convolve(motion, np.ones(5)/5, mode="same")
#     thresh = np.max(motion) * 0.2
#     mask = motion > thresh
#     if np.any(mask):
#         start_idx = int(np.argmax(mask))
#         end_idx = len(mask) - 1 - int(np.argmax(mask[::-1]))
#     else:
#         start_idx, end_idx = 0, len(frames)-1

#     # MediaPipe Pose가 있으면 follow-through 포함을 위해 조금 더 연장
#     if mp is not None:
#         end_idx = min(end_idx + 5, len(frames)-1)

#     start_idx = max(0, start_idx)
#     end_idx = min(len(frames)-1, end_idx)
#     if start_idx >= end_idx:
#         st.warning("스윙 구간 감지 실패. 전체 영상 슬로우 모션 처리합니다.")
#         start_idx, end_idx = 0, len(frames)-1

#     start_time = start_idx / fps
#     end_time = end_idx / fps
#     st.write(f"스윙 구간: **{start_time:.2f}s** → **{end_time:.2f}s** (프레임 {start_idx}→{end_idx})")

#     # (4) 슬로우 모션 처리
#     slow_factor = 0.5
#     output_path = "output_no_training.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     h, w = frames[0].shape[:2]

#     if mode.startswith("Local") and VideoFileClip:
#         clip = VideoFileClip(video_path)
#         pre_clip = clip.subclip(0, start_time) if start_idx > 0 else None
#         swing_clip = clip.subclip(start_time, end_time)
#         post_clip = clip.subclip(end_time, clip.duration) if end_idx < len(frames)-1 else None

#         slowed = swing_clip.fx(vfx.speedx, slow_factor)
#         parts = []
#         if pre_clip: parts.append(pre_clip)
#         parts.append(slowed)
#         if post_clip: parts.append(post_clip)

#         final_clip = concatenate_videoclips(parts)
#         temp_slow = "temp_slow.mp4"
#         final_clip.write_videofile(temp_slow, codec="libx264", fps=fps, audio=False)

#         cap2 = cv2.VideoCapture(temp_slow)
#         slow_frames = []
#         suc, fr = cap2.read()
#         while suc:
#             slow_frames.append(fr)
#             suc, fr = cap2.read()
#         cap2.release()
#     else:
#         slow_frames = []
#         dup_cnt = int(1/slow_factor)
#         dup_cnt = max(1, dup_cnt)
#         for i, fr in enumerate(frames):
#             if start_idx <= i <= end_idx:
#                 for _ in range(dup_cnt):
#                     slow_frames.append(fr.copy())
#             else:
#                 slow_frames.append(fr.copy())

#     st.write("공·클럽 헤드 트래킹을 시작합니다...")

#     # (5) MediaPipe Pose 초기화 (손목 좌표 이용)
#     pose = None
#     if mp is not None:
#         pose = mp.solutions.pose.Pose(static_image_mode=True)

#     raw_club = []  # 클럽 끝 좌표 기록 (u,v)
#     raw_ball = []  # 골프공 좌표 기록 (u,v)

#     # --- HoughCircles / HSV 마스크 파라미터 (골프공) ---
#     # 공은 화면 하단 중앙 30% 높이에 흰색 원이므로 HSV 범위를 타이트하게
#     lower_white = np.array([0, 0, 200], dtype=np.uint8)
#     upper_white = np.array([180, 50, 255], dtype=np.uint8)
#     hough_dp = 1.2
#     hough_minDist = 20
#     hough_param1 = 50
#     hough_param2 = 30
#     hough_minR = 4
#     hough_maxR = 15

#     # (6) Optical Flow / Shi-Tomasi 파라미터 (클럽 헤드)
#     feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
#     lk_params = dict(winSize=(15, 15), maxLevel=2,
#                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#     prev_gray_roi = None  # 클럽 헤드 추적용 이전 ROI 그레이스케일

#     for i, fr in enumerate(slow_frames):
#         club_pt = None
#         ball_pt = None

#         # (5-A) 클럽 헤드: 손목 인근 100×100 ROI에서 Optical Flow 추적
#         if pose is not None:
#             rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
#             res = pose.process(rgb)
#             if res.pose_landmarks:
#                 lm = res.pose_landmarks.landmark
#                 # 오른손목 (index=16)
#                 wx = int(lm[16].x * w)
#                 wy = int(lm[16].y * h)

#                 # ROI 정의 (frame 경계를 넘지 않도록)
#                 x0 = max(0, wx - 50)
#                 y0 = max(0, wy - 50)
#                 x1 = min(w, wx + 50)
#                 y1 = min(h, wy + 50)

#                 roi = fr[y0:y1, x0:x1]
#                 gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#                 if prev_gray_roi is not None and gray_roi.shape == prev_gray_roi.shape:
#                     # ROI 내에서 Shi-Tomasi 코너 검출
#                     p0 = cv2.goodFeaturesToTrack(prev_gray_roi, mask=None, **feature_params)
#                     if p0 is not None:
#                         # Lucas–Kanade Optical Flow 계산
#                         p1, stt, err = cv2.calcOpticalFlowPyrLK(prev_gray_roi, gray_roi, p0, None, **lk_params)
#                         # 이동 벡터 중 가장 큰 거리(벡터 크기) 포인트를 골라 클럽 헤드로 간주
#                         if p1 is not None:
#                             max_dist = 0
#                             best_pt = None
#                             for (new, old), s in zip(zip(p1.reshape(-1,2), p0.reshape(-1,2)), stt.flatten()):
#                                 if s == 1:
#                                     dx, dy = new[0]-old[0], new[1]-old[1]
#                                     dist = dx*dx + dy*dy
#                                     if dist > max_dist:
#                                         max_dist = dist
#                                         best_pt = new
#                             if best_pt is not None:
#                                 bx = int(best_pt[0]) + x0
#                                 by = int(best_pt[1]) + y0
#                                 club_pt = (bx, by)

#                 prev_gray_roi = gray_roi.copy()
#             else:
#                 prev_gray_roi = None
#         else:
#             prev_gray_roi = None

#         raw_club.append(club_pt)

#         # (5-B) 골프공: HSV 마스크 + HoughCircles
#         # 화면 하단 중앙 30% 범위 제한
#         y_start = int(h * 0.7)
#         x_start = int(w * 0.3)
#         x_end = int(w * 0.7)
#         roi2 = fr[y_start:h, x_start:x_end]

#         hsv = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, lower_white, upper_white)
#         # 모폴로지로 노이즈 제거
#         mask = cv2.medianBlur(mask, 5)

#         circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=hough_dp, minDist=hough_minDist,
#                                    param1=hough_param1, param2=hough_param2,
#                                    minRadius=hough_minR, maxRadius=hough_maxR)
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             # ROI 내부에서 y가 가장 큰 (화면 하단에 가까운) 원 선택
#             best = None
#             best_y = -1
#             for c in circles[0]:
#                 cx_r, cy_r, r = c
#                 if cy_r > best_y:
#                     best_y = cy_r
#                     best = (cx_r, cy_r, r)
#             if best is not None:
#                 cx, cy, r = best
#                 cx_full = cx + x_start
#                 cy_full = cy + y_start
#                 ball_pt = (cx_full, cy_full)
#         raw_ball.append(ball_pt)

#     if pose is not None:
#         pose.close()

#     # (6) 궤적 보간(스무딩)
#     def smooth_points(raw_pts, window=5):
#         filled = []
#         last = None
#         for p in raw_pts:
#             if p is not None:
#                 last = p
#                 filled.append(p)
#             else:
#                 filled.append(last)
#         smoothed = []
#         for i in range(len(filled)):
#             xs, ys = [], []
#             for j in range(i - window//2, i + window//2 + 1):
#                 if 0 <= j < len(filled) and filled[j] is not None:
#                     xs.append(filled[j][0])
#                     ys.append(filled[j][1])
#             if xs and ys:
#                 smoothed.append((int(sum(xs)/len(xs)), int(sum(ys)/len(ys))))
#             else:
#                 smoothed.append(None)
#         return smoothed

#     smooth_club = smooth_points(raw_club)
#     smooth_ball = smooth_points(raw_ball)

#     # (7) 궤적 오버레이
#     annotated = []
#     past_club = []
#     past_ball = []
#     for idx, fr in enumerate(slow_frames):
#         canvas = fr.copy()

#         # 클럽 궤적 (파란색)
#         if smooth_club[idx] is not None:
#             past_club.append(smooth_club[idx])
#         if len(past_club) >= 2:
#             pts = np.array(past_club, dtype=np.int32)
#             cv2.polylines(canvas, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
#         if past_club and past_club[-1] is not None:
#             cv2.circle(canvas, past_club[-1], 5, (255, 0, 0), -1)

#         # 공 궤적 (빨간색)
#         if smooth_ball[idx] is not None:
#             past_ball.append(smooth_ball[idx])
#         if len(past_ball) >= 2:
#             pts_b = np.array(past_ball, dtype=np.int32)
#             cv2.polylines(canvas, [pts_b], isClosed=False, color=(0, 0, 255), thickness=2)
#         if raw_ball[idx] is not None:
#             bx, by = raw_ball[idx]
#             cv2.circle(canvas, (bx, by), 5, (0, 0, 255), -1)

#         annotated.append(canvas)

#     # (8) 최종 영상 저장
#     out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
#     for fr in annotated:
#         out.write(fr)
#     out.release()

#     # (9) 즉시 재생 & 다운로드
#     if Path(output_path).exists():
#         st.video(output_path)
#         with open(output_path, "rb") as f:
#             data = f.read()
#         st.download_button(
#             "최종 영상 다운로드",
#             data=data,
#             file_name="swing_slowmo_no_training.mp4",
#             mime="video/mp4"
#         )
#     else:
#         st.error("결과 영상 생성에 실패했습니다.")


import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from moviepy.editor import VideoFileClip, ImageSequenceClip
import tempfile
import os
from io import BytesIO
import requests
import json

# 페이지 설정
st.set_page_config(
    page_title="골프 스윙 3D 분석기",
    page_icon="⛳",
    layout="wide"
)

st.title("⛳ 골프 스윙 3D 동영상 변환기")
st.markdown("스마트폰으로 촬영한 골프 스윙을 3D 카메라 트래킹 영상으로 변환합니다.")

# 사이드바 설정
st.sidebar.header("설정")
processing_mode = st.sidebar.selectbox(
    "처리 방식 선택",
    ["로컬 처리 (MediaPipe)", "외부 API (Google Cloud Vision)"]
)

slow_motion_factor = st.sidebar.slider(
    "슬로우 모션 배율",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.1
)

rotation_speed = st.sidebar.slider(
    "카메라 회전 속도",
    min_value=1,
    max_value=10,
    value=3
)

class GolfSwingAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_pose_landmarks(self, video_path):
        """비디오에서 포즈 랜드마크 추출"""
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        frames = []
        
        with st.progress(0) as progress_bar:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # 랜드마크 좌표 추출
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    landmarks_sequence.append(landmarks)
                    frames.append(frame)
                else:
                    landmarks_sequence.append(None)
                    frames.append(frame)
                
                current_frame += 1
                progress_bar.progress(current_frame / frame_count)
        
        cap.release()
        return landmarks_sequence, frames
    
    def detect_golf_swing_phases(self, landmarks_sequence):
        """골프 스윙 단계 자동 감지"""
        if not landmarks_sequence:
            return None, None
        
        # 손목 포인트 (15: 왼쪽 손목, 16: 오른쪽 손목)
        wrist_heights = []
        
        for landmarks in landmarks_sequence:
            if landmarks:
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                avg_height = (left_wrist[1] + right_wrist[1]) / 2
                wrist_heights.append(avg_height)
            else:
                wrist_heights.append(None)
        
        # None 값 제거 및 인덱스 매핑
        valid_heights = [(i, h) for i, h in enumerate(wrist_heights) if h is not None]
        
        if len(valid_heights) < 10:
            return None, None
        
        # Backswing-top 탐지 (최고점)
        heights_only = [h for _, h in valid_heights]
        backswing_top_idx = np.argmin(heights_only)  # y값이 작을수록 높은 위치
        
        # Address 시작점 (처음 10% 구간에서 안정된 지점)
        start_region = int(len(valid_heights) * 0.1)
        address_start = 0
        
        # Follow-through 끝점 (마지막 20% 구간에서 안정된 지점)
        end_region = int(len(valid_heights) * 0.8)
        follow_end = len(valid_heights) - 1
        
        return valid_heights[address_start][0], valid_heights[follow_end][0]
    
    def create_3d_visualization(self, landmarks_sequence, start_frame, end_frame):
        """3D 시각화 및 회전 애니메이션 생성"""
        if not landmarks_sequence:
            return None
        
        # 스윙 구간 추출
        swing_landmarks = landmarks_sequence[start_frame:end_frame+1]
        
        # 3D 플롯 생성
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 애니메이션 프레임 생성
        frames_data = []
        
        for frame_idx, landmarks in enumerate(swing_landmarks):
            if landmarks:
                # 주요 관절점만 선택
                key_points = [0, 1, 2, 5, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                x_coords = [landmarks[i][0] for i in key_points]
                y_coords = [landmarks[i][1] for i in key_points]
                z_coords = [landmarks[i][2] for i in key_points]
                
                frames_data.append((x_coords, y_coords, z_coords))
        
        return frames_data
    
    def create_rotating_video(self, frames_data, output_path, rotation_speed=3):
        """회전하는 3D 비디오 생성"""
        if not frames_data:
            return None
        
        temp_dir = tempfile.mkdtemp()
        image_files = []
        
        total_frames = len(frames_data) * 360 // rotation_speed
        
        with st.progress(0) as progress_bar:
            frame_count = 0
            
            for data_idx, (x_coords, y_coords, z_coords) in enumerate(frames_data):
                for angle in range(0, 360, rotation_speed):
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 3D 스캐터 플롯
                    ax.scatter(x_coords, y_coords, z_coords, c='red', s=50)
                    
                    # 연결선 그리기 (간단한 골격)
                    connections = [
                        (0, 1), (1, 2), (2, 5),  # 머리-어깨
                        (11, 12), (11, 13), (13, 15),  # 왼팔
                        (12, 14), (14, 16),  # 오른팔
                        (11, 23), (12, 24),  # 몸통
                        (23, 25), (25, 27),  # 왼다리
                        (24, 26), (26, 28)   # 오른다리
                    ]
                    
                    for start, end in connections:
                        if start < len(x_coords) and end < len(x_coords):
                            ax.plot([x_coords[start], x_coords[end]],
                                   [y_coords[start], y_coords[end]],
                                   [z_coords[start], z_coords[end]], 'b-')
                    
                    # 카메라 앵글 설정
                    ax.view_init(elev=20, azim=angle)
                    ax.set_xlim([0, 1])
                    ax.set_ylim([0, 1])
                    ax.set_zlim([-0.5, 0.5])
                    
                    # 이미지 저장
                    image_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                    plt.savefig(image_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    image_files.append(image_path)
                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
        
        # 이미지 시퀀스를 비디오로 변환
        clip = ImageSequenceClip(image_files, fps=24)
        clip.write_videofile(output_path, codec='libx264')
        
        # 임시 파일 정리
        for file in image_files:
            os.remove(file)
        os.rmdir(temp_dir)
        
        return output_path

class GoogleCloudVisionAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://vision.googleapis.com/v1"
    
    def analyze_video(self, video_bytes):
        """Google Cloud Vision API를 통한 비디오 분석"""
        # 실제 구현에서는 Google Cloud Video Intelligence API 사용
        st.warning("외부 API 연동은 데모용입니다. 실제 API 키와 설정이 필요합니다.")
        return None

def process_video_local(uploaded_file, slow_motion_factor, rotation_speed):
    """로컬 처리 함수"""
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        analyzer = GolfSwingAnalyzer()
        
        st.info("포즈 랜드마크를 추출 중입니다...")
        landmarks_sequence, frames = analyzer.extract_pose_landmarks(tmp_path)
        
        st.info("골프 스윙 구간을 감지 중입니다...")
        start_frame, end_frame = analyzer.detect_golf_swing_phases(landmarks_sequence)
        
        if start_frame is None or end_frame is None:
            st.error("골프 스윙 구간을 감지할 수 없습니다.")
            return None
        
        st.success(f"스윙 구간 감지 완료: {start_frame} ~ {end_frame} 프레임")
        
        st.info("3D 시각화를 생성 중입니다...")
        frames_data = analyzer.create_3d_visualization(landmarks_sequence, start_frame, end_frame)
        
        st.info("회전 비디오를 생성 중입니다...")
        output_path = tempfile.mktemp(suffix='.mp4')
        result_path = analyzer.create_rotating_video(frames_data, output_path, rotation_speed)
        
        # 슬로우 모션 적용
        if slow_motion_factor < 1.0:
            st.info("슬로우 모션을 적용 중입니다...")
            clip = VideoFileClip(result_path)
            slow_clip = clip.fx.speedx(slow_motion_factor)
            final_path = tempfile.mktemp(suffix='.mp4')
            slow_clip.write_videofile(final_path, codec='libx264')
            clip.close()
            slow_clip.close()
            os.remove(result_path)
            result_path = final_path
        
        return result_path
        
    finally:
        os.unlink(tmp_path)

def process_video_api(uploaded_file, api_key):
    """외부 API 처리 함수"""
    if not api_key:
        st.error("API 키를 입력해주세요.")
        return None
    
    api_client = GoogleCloudVisionAPI(api_key)
    result = api_client.analyze_video(uploaded_file.read())
    return result

# 메인 인터페이스
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 비디오 업로드")
    uploaded_file = st.file_uploader(
        "골프 스윙 동영상을 업로드하세요",
        type=['mp4', 'avi', 'mov'],
        help="MP4, AVI, MOV 형식을 지원합니다."
    )

with col2:
    st.header("⚙️ 처리 옵션")
    
    if processing_mode == "외부 API (Google Cloud Vision)":
        api_key = st.text_input(
            "Google Cloud API 키",
            type="password",
            help="Google Cloud Vision API 키를 입력하세요"
        )

if uploaded_file is not None:
    st.header("📊 결과")
    
    # 원본 비디오 미리보기
    st.subheader("원본 비디오")
    st.video(uploaded_file)
    
    # 처리 시작 버튼
    if st.button("🚀 3D 변환 시작", type="primary"):
        with st.spinner("비디오를 처리 중입니다..."):
            if processing_mode == "로컬 처리 (MediaPipe)":
                result_path = process_video_local(
                    uploaded_file, 
                    slow_motion_factor, 
                    rotation_speed
                )
            else:
                result_path = process_video_api(uploaded_file, api_key)
            
            if result_path:
                st.success("변환 완료!")
                
                # 결과 비디오 표시
                st.subheader("3D 회전 결과")
                with open(result_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 결과 비디오 다운로드",
                    data=video_bytes,
                    file_name=f"golf_swing_3d_{uploaded_file.name}",
                    mime="video/mp4"
                )
                
                # 임시 파일 정리
                os.remove(result_path)
            else:
                st.error("비디오 처리에 실패했습니다.")

# 사용법 안내
with st.expander("📖 사용법 안내"):
    st.markdown("""
    ### 사용 단계
    1. **비디오 업로드**: 골프 스윙이 포함된 MP4 동영상을 업로드하세요
    2. **처리 방식 선택**: 로컬 처리 또는 외부 API 중 선택하세요
    3. **옵션 설정**: 슬로우 모션 배율과 카메라 회전 속도를 조정하세요
    4. **변환 시작**: '3D 변환 시작' 버튼을 클릭하세요
    
    ### 팁
    - 골프 스윙이 명확히 보이는 각도에서 촬영된 영상을 사용하세요
    - 배경이 복잡하지 않은 환경에서 촬영하면 더 정확한 결과를 얻을 수 있습니다
    - 로컬 처리는 개인정보 보호에 유리하지만 처리 시간이 더 오래 걸릴 수 있습니다
    """)

# 기술 정보
with st.expander("🔧 기술 정보"):
    st.markdown("""
    ### 사용된 기술
    - **MediaPipe**: Google의 포즈 추정 라이브러리
    - **OpenCV**: 비디오 처리 및 컴퓨터 비전
    - **Matplotlib**: 3D 시각화 및 애니메이션
    - **MoviePy**: 비디오 편집 및 효과
    - **Streamlit**: 웹 애플리케이션 프레임워크
    
    ### 시스템 요구사항
    - M3 맥 또는 동급 CPU
    - Python 3.8 이상
    - 8GB 이상 RAM 권장
    """)
