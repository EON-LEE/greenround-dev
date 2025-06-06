import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SwingVisualizer3D:
    def __init__(self):
        """3D 시각화를 위한 초기화"""
        self.connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
        
        self.colors = {
            'markers': '#2E86C1',
            'connections': '#3498DB',
            'trajectory': '#E74C3C',
            'plane': '#2ECC71'
        }
        
    def create_pose_plot(self, frame_data: Dict, show_trajectory: bool = False,
                        trajectory_data: Optional[List[Dict]] = None) -> go.Figure:
        """3D 포즈 플롯 생성"""
        try:
            fig = go.Figure()
            
            # 랜드마크 점 추가
            self._add_landmarks(fig, frame_data['landmarks'])
            
            # 스켈레톤 연결선 추가
            self._add_connections(fig, frame_data['landmarks'])
            
            # 궤적 추가 (옵션)
            if show_trajectory and trajectory_data:
                self._add_trajectory(fig, trajectory_data)
            
            # 스윙 플레인 추가
            self._add_swing_plane(fig, frame_data['landmarks'])
            
            # 레이아웃 설정
            self._set_layout(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pose plot: {str(e)}")
            return go.Figure()
            
    def _add_landmarks(self, fig: go.Figure, landmarks: Dict):
        """랜드마크 점 추가"""
        try:
            for name, point in landmarks.items():
                fig.add_trace(go.Scatter3d(
                    x=[point[0]], y=[point[1]], z=[point[2]],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=8,
                        color=self.colors['markers'],
                        symbol='circle'
                    ),
                    hoverinfo='name+text',
                    text=f'[{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]'
                ))
                
        except Exception as e:
            logger.error(f"Error adding landmarks: {str(e)}")
            
    def _add_connections(self, fig: go.Figure, landmarks: Dict):
        """스켈레톤 연결선 추가"""
        try:
            for start, end in self.connections:
                if start in landmarks and end in landmarks:
                    start_point = landmarks[start]
                    end_point = landmarks[end]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[start_point[0], end_point[0]],
                        y=[start_point[1], end_point[1]],
                        z=[start_point[2], end_point[2]],
                        mode='lines',
                        line=dict(
                            color=self.colors['connections'],
                            width=5
                        ),
                        showlegend=False
                    ))
                    
        except Exception as e:
            logger.error(f"Error adding connections: {str(e)}")
            
    def _add_trajectory(self, fig: go.Figure, trajectory_data: List[Dict]):
        """스윙 궤적 추가"""
        try:
            # 클럽 헤드(오른쪽 손목) 궤적
            x_coords = []
            y_coords = []
            z_coords = []
            
            for frame in trajectory_data:
                point = frame['landmarks']['right_wrist']
                x_coords.append(point[0])
                y_coords.append(point[1])
                z_coords.append(point[2])
                
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                name='Swing Path',
                line=dict(
                    color=self.colors['trajectory'],
                    width=3
                ),
                opacity=0.7
            ))
            
        except Exception as e:
            logger.error(f"Error adding trajectory: {str(e)}")
            
    def _add_swing_plane(self, fig: go.Figure, landmarks: Dict):
        """스윙 플레인 추가"""
        try:
            # 어깨 선과 힙 선을 이용하여 스윙 플레인 계산
            shoulder_vector = np.array(landmarks['right_shoulder']) - np.array(landmarks['left_shoulder'])
            hip_vector = np.array(landmarks['right_hip']) - np.array(landmarks['left_hip'])
            
            # 플레인의 법선 벡터 계산
            normal = np.cross(shoulder_vector, hip_vector)
            normal = normal / np.linalg.norm(normal)
            
            # 플레인 생성을 위한 그리드 포인트 생성
            center = np.mean([
                landmarks['right_shoulder'],
                landmarks['left_shoulder'],
                landmarks['right_hip'],
                landmarks['left_hip']
            ], axis=0)
            
            grid_size = 1.0
            xx, zz = np.meshgrid(
                np.linspace(center[0] - grid_size, center[0] + grid_size, 10),
                np.linspace(center[2] - grid_size, center[2] + grid_size, 10)
            )
            
            # 평면 방정식을 이용하여 y 좌표 계산
            yy = (-normal[0] * (xx - center[0]) - normal[2] * (zz - center[2])) / normal[1] + center[1]
            
            # 플레인 추가
            fig.add_trace(go.Surface(
                x=xx,
                y=yy,
                z=zz,
                opacity=0.3,
                showscale=False,
                colorscale=[[0, self.colors['plane']], [1, self.colors['plane']]],
                name='Swing Plane'
            ))
            
        except Exception as e:
            logger.error(f"Error adding swing plane: {str(e)}")
            
    def _set_layout(self, fig: go.Figure):
        """플롯 레이아웃 설정"""
        try:
            fig.update_layout(
                scene=dict(
                    aspectmode='data',
                    camera=dict(
                        up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    xaxis=dict(title='X'),
                    yaxis=dict(title='Y'),
                    zaxis=dict(title='Z')
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
        except Exception as e:
            logger.error(f"Error setting layout: {str(e)}")
            
    def create_trajectory_animation(self, frames_data: List[Dict]) -> go.Figure:
        """스윙 궤적 애니메이션 생성"""
        try:
            fig = go.Figure()
            
            # 프레임 데이터 준비
            frames = []
            for i, frame_data in enumerate(frames_data):
                frame = go.Frame(
                    data=self._create_frame_data(frame_data, frames_data[:i+1]),
                    name=f'frame_{i}'
                )
                frames.append(frame)
            
            # 초기 프레임 설정
            initial_data = self._create_frame_data(frames_data[0], [frames_data[0]])
            for trace in initial_data:
                fig.add_trace(trace)
            
            # 프레임 추가
            fig.frames = frames
            
            # 애니메이션 컨트롤 추가
            self._add_animation_controls(fig, len(frames))
            
            # 레이아웃 설정
            self._set_layout(fig)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trajectory animation: {str(e)}")
            return go.Figure()
            
    def _create_frame_data(self, frame_data: Dict, trajectory_data: List[Dict]) -> List[go.Scatter3d]:
        """각 프레임의 데이터 생성"""
        try:
            frame_traces = []
            
            # 랜드마크와 연결선 추가
            landmarks = frame_data['landmarks']
            
            # 랜드마크 점
            for name, point in landmarks.items():
                frame_traces.append(go.Scatter3d(
                    x=[point[0]], y=[point[1]], z=[point[2]],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=8,
                        color=self.colors['markers'],
                        symbol='circle'
                    )
                ))
            
            # 연결선
            for start, end in self.connections:
                if start in landmarks and end in landmarks:
                    start_point = landmarks[start]
                    end_point = landmarks[end]
                    frame_traces.append(go.Scatter3d(
                        x=[start_point[0], end_point[0]],
                        y=[start_point[1], end_point[1]],
                        z=[start_point[2], end_point[2]],
                        mode='lines',
                        line=dict(
                            color=self.colors['connections'],
                            width=5
                        ),
                        showlegend=False
                    ))
            
            # 궤적 추가
            x_coords = [frame['landmarks']['right_wrist'][0] for frame in trajectory_data]
            y_coords = [frame['landmarks']['right_wrist'][1] for frame in trajectory_data]
            z_coords = [frame['landmarks']['right_wrist'][2] for frame in trajectory_data]
            
            frame_traces.append(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                name='Swing Path',
                line=dict(
                    color=self.colors['trajectory'],
                    width=3
                ),
                opacity=0.7
            ))
            
            return frame_traces
            
        except Exception as e:
            logger.error(f"Error creating frame data: {str(e)}")
            return []
            
    def _add_animation_controls(self, fig: go.Figure, n_frames: int):
        """애니메이션 컨트롤 추가"""
        try:
            fig.update_layout(
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        buttons=[
                            dict(
                                label='Play',
                                method='animate',
                                args=[None, dict(
                                    frame=dict(duration=50, redraw=True),
                                    fromcurrent=True,
                                    mode='immediate'
                                )]
                            ),
                            dict(
                                label='Pause',
                                method='animate',
                                args=[[None], dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )]
                            )
                        ],
                        direction='left',
                        pad=dict(r=10, t=10),
                        x=0.1,
                        y=0,
                        xanchor='right',
                        yanchor='top'
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        yanchor='top',
                        xanchor='left',
                        currentvalue=dict(
                            font=dict(size=20),
                            prefix='Frame: ',
                            visible=True,
                            xanchor='right'
                        ),
                        pad=dict(b=10, t=50),
                        len=0.9,
                        x=0.1,
                        y=0,
                        steps=[
                            dict(
                                method='animate',
                                args=[[f'frame_{i}'], dict(
                                    frame=dict(duration=50, redraw=True),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )],
                                label=str(i)
                            )
                            for i in range(n_frames)
                        ]
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error adding animation controls: {str(e)}") 