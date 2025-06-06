# file: app.py

import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
import torch
from moviepy.editor import VideoFileClip, vfx, ImageSequenceClip
from mediapipe.python.solutions import pose as mp_pose


# ──────────────────────────────────────────────────────────────────────────────
# 1. 스윙 구간 감지 (MediaPipe Pose)
def detect_swing_segment(video_path):
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    start, end = None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    swing_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # MediaPipe로 왼손 위치(or 오른손) 추적하여 스윙 시점 검출
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            if wrist.visibility > 0.7:
                if not swing_detected:
                    start = idx
                    swing_detected = True
                end = idx
        idx += 1

    cap.release()
    # 시작/끝 프레임을 초단위로 반환
    return (start / fps) if start is not None else 0.0, (end / fps) if end is not None else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. 슬로우모션 생성 (MoviePy)
def extract_and_slow_motion(video_path, start, end):
    clip = VideoFileClip(video_path).subclip(start, end)
    slowed = clip.fx(vfx.speedx, 0.25)  # 1/4속도, 필요하면 0.3로도 변경
    return slowed


# ──────────────────────────────────────────────────────────────────────────────
# 3. 깊이맵 → 포인트 클라우드로 변환 (단일 프레임 기준)
def depth_to_point_cloud(image, depth):
    """
    image: PIL.Image (RGB)
    depth: 2D numpy array (float32), 크기는 image와 동일
    """
    h, w = depth.shape
    fx = fy = 1.0
    cx, cy = w / 2.0, h / 2.0

    # (x, y) 좌표 그리드 생성
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth

    # 각 픽셀을 3D 좌표로 변환
    x3 = (x - cx) * z / fx
    y3 = (y - cy) * z / fy
    xyz = np.stack((x3, y3, z), axis=-1).reshape(-1, 3)  # (H*W, 3)

    # 컬러 정보
    rgb = np.asarray(image).reshape(-1, 3) / 255.0  # (H*W, 3)

    return xyz, rgb


# ──────────────────────────────────────────────────────────────────────────────
# 4. Z-버퍼 렌더러로 360° 회전 영상 생성
def render_orbit_for_slowclip(slow_clip, save_dir):
    """
    slow_clip: MoviePy VideoFileClip (슬로우모션 전체 구간)
    save_dir: 임시 저장 디렉토리
    """
    # (a) 슬로우모션의 각 프레임을 PIL.Image로 추출
    frames = []
    for frame in slow_clip.iter_frames(fps=15):  # 슬로우 클립 FPS를 15로 고정
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    n_frames = len(frames)

    # (b) 각 프레임 별 깊이맵 생성 (MiDaS)
    st.write(f"🔍 총 {n_frames} 프레임에 대해 깊이맵 계산 중...")
    midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas_model.eval()
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    depth_maps = []
    for img in frames:
        inp = midas_transform(np.array(img)).to("cpu")
        with torch.no_grad():
            pred = midas_model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.size[::-1], mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
        depth_maps.append(pred)

    # (c) 각 프레임마다 3D 포인트 클라우드 생성 및 회전 렌더링
    st.write("🚀 3D 회전 하이라이트 프레임 생성 중 (Z-버퍼 렌더러)...")
    all_orbit_frames = []  # 최종 전체 3D 회전 프레임을 담을 리스트

    canvas_size = 256  # 결과 영상 해상도
    num_angles = 60    # 각 프레임 당 60장 시점 생성 (360도 / 60 = 6도 단위)

    for idx, (img_pil, depth) in enumerate(zip(frames, depth_maps)):
        # (1) 다운샘플 (256×256 성능 한계 고려)
        img_small = img_pil.resize((256, 256))
        depth_ds = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)

        # (2) 포인트 클라우드 생성
        pts, cols = depth_to_point_cloud(img_small, depth_ds)
        center = pts.mean(axis=0)
        pts_centered = pts - center  # 메쉬/포인트 클라우드 중심 정렬

        # (3) 각 시점별 Z-버퍼 렌더링
        #    -> (H*W)개 포인트 이용해 360도 orbit
        for angle in np.linspace(0, 2*np.pi, num_angles, endpoint=False):
            # 회전 행렬 (Y축 주위)
            ca, sa = np.cos(angle), np.sin(angle)
            R = np.array([[ ca, 0, sa],
                          [  0, 1,  0],
                          [-sa, 0, ca]])
            pts_rot = pts_centered.dot(R.T)  # (N,3)

            # 원근 투영 (카메라는 z축 뒤쪽에 위치)
            z_cam = pts_rot[:, 2] + 3.0   # 모두 +3.0 밀어 카메라 앞쪽으로
            eps = 1e-6
            u = (pts_rot[:, 0] / (z_cam + eps) * (canvas_size/2)) + (canvas_size/2)
            v = (pts_rot[:, 1] / (z_cam + eps) * (canvas_size/2)) + (canvas_size/2)
            u = np.round(u).astype(int)
            v = np.round(v).astype(int)

            # Z-버퍼 초기화
            zbuf = np.full((canvas_size, canvas_size), np.inf)
            cbuf = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)

            # 포인트마다 Z-버퍼 테스트 후 색상 저장
            for i_pt in range(pts_centered.shape[0]):
                ui, vi, zi = u[i_pt], v[i_pt], z_cam[i_pt]
                if 0 <= ui < canvas_size and 0 <= vi < canvas_size:
                    if zi < zbuf[vi, ui]:
                        zbuf[vi, ui] = zi
                        cbuf[vi, ui, :] = cols[i_pt]

            frame_np = (np.clip(cbuf, 0, 1) * 255).astype(np.uint8)
            all_orbit_frames.append(frame_np)

        st.write(f"  ▶ Frame {idx+1}/{n_frames} 처리 완료")

    # (d) 최종 3D 회전 하이라이트 비디오 작성
    orbit_video_path = os.path.join(tempfile.gettempdir(), "orbit_highlight.mp4")
    ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in all_orbit_frames], fps=15).write_videofile(orbit_video_path, codec="libx264")

    return orbit_video_path


# ──────────────────────────────────────────────────────────────────────────────
# 5. Streamlit 메인 함수
def main():
    st.title("🏌️‍♂️ 골프 스윙 하이라이트 생성기 (NumPy Z-버퍼 3D 회전)")

    uploaded = st.file_uploader("골프 스윙 동영상을 업로드하세요 (.mp4)", type="mp4")
    if not uploaded:
        st.info("왼쪽 상단에서 동영상을 업로드해주세요.")
        return

    # (a) 업로드된 파일 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded.read())
        video_path = tfile.name

    st.video(video_path)

    # (b) 스윙 구간 감지
    st.info("🔍 스윙 구간 감지 중...")
    start, end = detect_swing_segment(video_path)
    st.success(f"✅ 스윙 구간: {start:.2f}초 ~ {end:.2f}초")

    # (c) 슬로우 모션 생성
    st.info("🐢 슬로우 모션 영상 생성 중...")
    slow_clip = extract_and_slow_motion(video_path, start, end)
    slow_path = os.path.join(tempfile.gettempdir(), "slow_motion.mp4")
    slow_clip.write_videofile(slow_path, codec="libx264", audio=False)
    st.video(slow_path)

    # (d) 3D 회전 하이라이트 생성
    st.info("🚀 3D 회전 하이라이트 렌더링 중... (상황에 따라 수 분 소요될 수 있습니다)")
    orbit_path = render_orbit_for_slowclip(slow_clip, tempfile.gettempdir())
    st.success("🎉 3D 회전 하이라이트 완성!")

    st.video(orbit_path)
    with open(orbit_path, "rb") as f:
        st.download_button("🎬 하이라이트 영상 다운로드", f, file_name="highlight_output.mp4")


if __name__ == "__main__":
    main()
