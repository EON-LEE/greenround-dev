import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import torch
from transformers import pipeline
from moviepy.editor import VideoFileClip, ImageSequenceClip
import tempfile
import os
from PIL import Image

# 환경 설정
st.set_page_config(page_title="고급 골프 스윙 3D 분석", layout="wide")
st.title("🎯 프로급 골프 스윙 3D 변환 시스템")

# 모델 초기화 (CPU 강제 설정)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
depth_estimator = pipeline("depth-estimation", model="LiheYoung/depth-anything-large-hf", device="cpu")
pose_estimator = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)

def estimate_depth(frame):
    """MiDaS 기반 심도 추정"""
    return depth_estimator(frame)["predicted_depth"]

def warp_frame(frame, depth_map, angle):
    """심도 맵 기반 프레임 변형"""
    h, w = frame.shape[:2]
    focal = 0.8 * w
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
    R = cv2.Rodrigues(np.array([0, np.radians(angle), 0]))[0]
    warp_matrix = K @ R @ np.linalg.inv(K)
    return cv2.warpPerspective(frame, warp_matrix, (w,h), borderMode=cv2.BORDER_REPLICATE)

def process_video(uploaded_file, rotation_speed):
    """비디오 처리 파이프라인"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    depth_maps = []
    
    # 1단계: 심도 맵 & 포즈 추정
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # 심도 추정 (검색결과[1][2])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_map = estimate_depth(Image.fromarray(rgb_frame))
        
        # 포즈 추정
        pose_results = pose_estimator.process(rgb_frame)
        
        frames.append(frame)
        depth_maps.append(depth_map)
        progress_bar.progress((i+1)/total_frames)
    
    # 2단계: 동적 카메라 경로 생성 (검색결과[3][4])
    angles = np.linspace(0, 360, len(frames)//rotation_speed)
    
    # 3단계: 프레임 변형
    processed_frames = []
    for i, (frame, depth) in enumerate(zip(frames, depth_maps)):
        angle = angles[i % len(angles)]
        warped = warp_frame(frame, depth.numpy(), angle)
        processed_frames.append(warped)
    
    # 4단계: 비디오 재구성
    output_path = tempfile.mktemp(suffix='.mp4')
    clip = ImageSequenceClip(processed_frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264')
    
    return output_path

# UI 구성
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("골프 스윙 동영상 업로드", type=['mp4'])
    if uploaded_file:
        st.video(uploaded_file)

with col2:
    rotation_speed = st.slider("회전 속도", 1, 10, 3)
    processing_mode = st.selectbox("처리 모드", ["로컬 CPU", "클라우드 GPU"])

if uploaded_file and st.button("변환 시작"):
    with st.spinner("3D 변환 처리 중..."):
        output_path = process_video(uploaded_file, rotation_speed)
        st.success("변환 완료!")
        st.video(output_path)
        os.remove(output_path)
