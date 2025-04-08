"""
Core pose processing functionality for the ShadowCoach system
"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..utils.logging_utils import logger, timed

class PoseProcessor:
    """
    Class for processing videos and extracting pose data
    """
    def __init__(self):
        """Initialize the pose processor with MediaPipe"""
        logger.info("Initializing PoseProcessor...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # More accurate model
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmarks for boxing analysis
        self.boxing_landmarks = [
            'nose', 
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        
        # Mapping from landmark names to MediaPipe indices
        self.landmark_indices = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        logger.info("PoseProcessor initialized successfully")
    
    @timed
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     visualize: bool = True) -> List[Dict[str, Dict[str, float]]]:
        """
        Process a video and extract pose data
        
        Args:
            video_path: Path to the video file
            output_path: Optional path to save the processed video
            visualize: Whether to visualize the pose landmarks
            
        Returns:
            A time series of landmark positions
        """
        logger.info(f"Starting to process video: {video_path}")
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []
            
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Output video writer
        if output_path:
            logger.info(f"Setting up output video: {output_path}")
            Path(output_path).parent.mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        poses = []
        frame_idx = 0
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Extract landmarks if detected
            if results.pose_landmarks:
                # Create a dictionary of landmark positions
                pose_data = {}
                for name, idx in self.landmark_indices.items():
                    landmark = results.pose_landmarks.landmark[idx]
                    pose_data[name] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
                poses.append(pose_data)
                
                # Draw landmarks if visualizing
                if visualize and output_path:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Add frame number
                    cv2.putText(
                        frame, 
                        f"Frame: {frame_idx}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            
            # Write to output video
            if output_path:
                out.write(frame)
                
            frame_idx += 1
            
            # Log progress periodically
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
        
        # Release resources
        video.release()
        if output_path:
            out.release()
            
        logger.info(f"Video processing complete. Extracted {len(poses)} poses from {frame_idx} frames.")
        
        # Apply smoothing to reduce noise
        if poses:
            poses = self._smooth_pose_data(poses)
            
        return poses
    
    @timed
    def _smooth_pose_data(self, poses: List[Dict[str, Dict[str, float]]], 
                         window: int = 15, poly: int = 3) -> List[Dict[str, Dict[str, float]]]:
        """
        Apply smoothing to the pose data to reduce noise
        
        Args:
            poses: The pose data to smooth
            window: The window size for the smoothing filter
            poly: The polynomial order for the smoothing filter
            
        Returns:
            Smoothed pose data
        """
        if len(poses) < window:
            logger.warning(f"Not enough frames for smoothing: {len(poses)} < {window}")
            return poses
            
        # Make window odd if it's even
        if window % 2 == 0:
            window += 1
            
        logger.info(f"Smoothing pose data with window={window}, poly={poly}")
        
        # Create a deep copy of the pose data
        smoothed_poses = [{k: v.copy() for k, v in pose.items()} for pose in poses]
        
        # For each landmark and dimension, apply Savitzky-Golay filter
        for landmark in self.boxing_landmarks:
            for dim in ['x', 'y', 'z']:
                # Extract the values
                values = [pose[landmark][dim] for pose in poses if landmark in pose]
                
                if len(values) >= window:
                    # Apply smoothing
                    from scipy.signal import savgol_filter
                    smoothed_values = savgol_filter(values, window, poly)
                    
                    # Update the smoothed poses
                    for i, pose in enumerate(smoothed_poses):
                        if landmark in pose:
                            pose[landmark][dim] = smoothed_values[i]
        
        logger.info("Pose data smoothing complete")
        return smoothed_poses
