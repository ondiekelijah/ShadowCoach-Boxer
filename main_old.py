import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from pathlib import Path
import pickle
import time
import logging
import functools
import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('ShadowCoach')

# Dictionary to store function execution times
function_timings = {}

def timed(func):
    """Decorator to measure and log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        logger.debug(f"Starting {func_name}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store timing information
        if func_name not in function_timings:
            function_timings[func_name] = []
        function_timings[func_name].append(execution_time)
        
        logger.debug(f"Completed {func_name} in {execution_time:.2f} seconds")
        return result
    return wrapper

def generate_unique_filename(base_path, prefix="", suffix=""):
    """Generate a unique filename with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}{suffix}" if prefix else f"{timestamp}{suffix}"
    return os.path.join(base_path, filename)

class BoxingAnalyzer:
    def __init__(self):
        logger.info("Initializing BoxingAnalyzer...")
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
        
        # Reference data storage
        self.references = {}
        logger.info("BoxingAnalyzer initialized successfully")
    
    @timed
    def process_video(self, video_path, output_path=None, visualize=True):
        """
        Process a video and extract pose data
        Returns: A time series of landmark positions
        """
        start_time = time.time()
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
        
        poses = []
        
        # Process frames
        frame_idx = 0
        last_log_time = time.time()
        log_interval = 2.0  # Log progress every 2 seconds
        
        logger.info("Beginning frame processing...")
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract key landmarks
                frame_landmarks = {}
                for name in self.boxing_landmarks:
                    idx = self.landmark_indices[name]
                    landmark = results.pose_landmarks.landmark[idx]
                    # Convert to pixel coordinates
                    frame_landmarks[name] = {
                        'x': landmark.x * width,
                        'y': landmark.y * height,
                        'z': landmark.z * width,  # Scale Z similarly to X
                        'visibility': landmark.visibility
                    }
                
                poses.append(frame_landmarks)
                
                # Visualize if requested
                if visualize:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Add frame information
                    cv2.putText(
                        frame, 
                        f'Frame: {frame_idx}', 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    
                    if output_path:
                        out.write(frame)
            
            frame_idx += 1
            
            # Log progress at intervals
            current_time = time.time()
            if current_time - last_log_time > log_interval:
                elapsed = current_time - start_time
                progress = frame_idx / frame_count * 100
                frames_per_sec = frame_idx / elapsed
                remaining_time = (frame_count - frame_idx) / frames_per_sec if frames_per_sec > 0 else 0
                
                logger.info(f"Processed {frame_idx}/{frame_count} frames ({progress:.1f}%) - "
                           f"{frames_per_sec:.1f} FPS - Est. remaining: {remaining_time:.1f}s")
                last_log_time = current_time
        
        video.release()
        if output_path:
            out.release()
            
        total_time = time.time() - start_time
        logger.info(f"Video processing complete: {frame_idx} frames in {total_time:.2f}s ({frame_idx/total_time:.1f} FPS)")
        
        if len(poses) > 0:
            logger.info("Applying smoothing to pose data...")
            return self._smooth_pose_data(poses)
        else:
            logger.warning("No poses detected in the video!")
            return []
    
    @timed
    def _smooth_pose_data(self, poses, window=15, poly=3):
        """Apply smoothing to the pose data to reduce noise"""
        if len(poses) < window:
            logger.warning(f"Not enough frames to smooth: {len(poses)} < {window}")
            return poses  # Not enough frames to smooth
            
        logger.info(f"Smoothing pose data with window size {window}...")
        smoothed_poses = []
        
        # Extract time series for each coordinate
        time_series = {name: {'x': [], 'y': [], 'z': []} for name in self.boxing_landmarks}
        visibility = {name: [] for name in self.boxing_landmarks}
        
        for frame in poses:
            for name in self.boxing_landmarks:
                if name in frame:
                    time_series[name]['x'].append(frame[name]['x'])
                    time_series[name]['y'].append(frame[name]['y'])
                    time_series[name]['z'].append(frame[name]['z'])
                    visibility[name].append(frame[name]['visibility'])
        
        # Apply Savitzky-Golay filter to each coordinate
        for name in self.boxing_landmarks:
            for coord in ['x', 'y', 'z']:
                if len(time_series[name][coord]) >= window:
                    time_series[name][coord] = savgol_filter(
                        time_series[name][coord], window, poly
                    ).tolist()
        
        # Reconstruct frame-by-frame data
        for i in range(len(poses)):
            frame = {}
            for name in self.boxing_landmarks:
                if i < len(time_series[name]['x']):
                    frame[name] = {
                        'x': time_series[name]['x'][i],
                        'y': time_series[name]['y'][i],
                        'z': time_series[name]['z'][i],
                        'visibility': visibility[name][i]
                    }
            smoothed_poses.append(frame)
            
        logger.info(f"Smoothing complete: {len(smoothed_poses)} frames processed")
        return smoothed_poses
    
    @timed
    def save_reference(self, name, pose_data, metadata=None):
        """Save a reference model"""
        logger.info(f"Saving reference model '{name}' with {len(pose_data)} frames")
        self.references[name] = {
            'poses': pose_data,
            'metadata': metadata or {}
        }
        
    @timed
    def save_references_to_file(self, path):
        """Save all references to a file"""
        logger.info(f"Saving {len(self.references)} reference models to {path}")
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.references, f)
        logger.info(f"References saved successfully to {path}")
    
    @timed
    def load_references_from_file(self, path):
        """Load references from a file"""
        if not Path(path).exists():
            logger.error(f"Reference file not found: {path}")
            return False
            
        logger.info(f"Loading reference models from {path}")
        try:
            with open(path, 'rb') as f:
                self.references = pickle.load(f)
            logger.info(f"Loaded {len(self.references)} reference models")
            return True
        except Exception as e:
            logger.error(f"Error loading references: {str(e)}")
            return False
    
    @timed
    def analyze_jab(self, pose_data, reference_name=None):
        """
        Analyze jab technique by comparing to reference
        If reference_name is None, try to auto-detect jabs
        Returns analysis metrics
        """
        logger.info(f"Analyzing jab technique from {len(pose_data)} frames of pose data")
        
        # Extract features and detect jabs
        jab_analysis = self._detect_jabs(pose_data)
        
        # If no jabs detected or features couldn't be extracted, return empty results
        if not jab_analysis:
            logger.warning("Could not extract features for jab analysis")
            return {
                'arm_angles': [],
                'jab_speeds': [],
                'jabs': [],
                'avg_metrics': {
                    'avg_duration': 0,
                    'avg_max_speed': 0,
                    'avg_max_extension': None
                },
                'reference_comparison': None
            }
        
        # Compare to reference if provided
        reference_comparison = None
        if reference_name and reference_name in self.references:
            reference_comparison = self._compare_to_reference(
                pose_data, 
                jab_analysis, 
                reference_name
            )
        
        return {
            'arm_angles': jab_analysis['arm_angles'],
            'jab_speeds': jab_analysis['jab_speeds'],
            'jabs': jab_analysis['jabs'],
            'avg_metrics': jab_analysis['avg_metrics'],
            'reference_comparison': reference_comparison
        }
    
    def _detect_jabs(self, pose_data):
        """Extract features and detect jabs from pose data"""
        # Extract normalized features that are orientation-invariant
        normalized_features = self._extract_normalized_features(pose_data)
        
        if not normalized_features:
            return None
        
        # Extract arm angles from normalized features
        right_arm_angles = [features['r_angle'] for features in normalized_features]
        
        # Calculate jab speeds using normalized arm extension changes
        jab_speeds = self._calculate_jab_speeds(normalized_features)
        
        # Detect jabs using adaptive thresholding
        potential_jabs = self._identify_potential_jabs(normalized_features, jab_speeds, right_arm_angles)
        
        logger.info(f"Detected {len(potential_jabs)} potential jabs")
        
        # Calculate average metrics across all jabs
        avg_metrics = self._calculate_average_metrics(potential_jabs)
        
        return {
            'arm_angles': right_arm_angles,
            'jab_speeds': jab_speeds,
            'jabs': potential_jabs,
            'avg_metrics': avg_metrics
        }
    
    def _calculate_jab_speeds(self, normalized_features):
        """Calculate jab speeds from normalized features"""
        jab_speeds = []
        for i in range(1, len(normalized_features)):
            prev_features = normalized_features[i-1]
            curr_features = normalized_features[i]
            
            # Calculate change in arm extension (normalized by shoulder width)
            r_extension_change = abs(curr_features['r_extension'] - prev_features['r_extension'])
            l_extension_change = abs(curr_features['l_extension'] - prev_features['l_extension'])
            
            # Calculate change in arm direction
            r_direction_change = np.sqrt(
                (curr_features['r_direction_x'] - prev_features['r_direction_x'])**2 +
                (curr_features['r_direction_y'] - prev_features['r_direction_y'])**2
            )
            
            # Determine dominant hand based on which arm moves more
            r_changes = [f['r_extension_change'] for f in normalized_features if 'r_extension_change' in f]
            l_changes = [f['l_extension_change'] for f in normalized_features if 'l_extension_change' in f]
            
            if r_changes and l_changes and np.mean(r_changes) > np.mean(l_changes):
                # Right hand dominant
                speed = r_extension_change * 0.7 + r_direction_change * 0.3
            else:
                # Left hand dominant
                speed = l_extension_change * 0.7 + r_direction_change * 0.3
                
            jab_speeds.append(speed)
            
            # Store these for the next frame
            normalized_features[i]['r_extension_change'] = r_extension_change
            normalized_features[i]['l_extension_change'] = l_extension_change
        
        # Add a zero at the beginning to match the length of pose_data
        jab_speeds.insert(0, 0)
        return jab_speeds
    
    def _identify_potential_jabs(self, normalized_features, jab_speeds, arm_angles):
        """Identify potential jabs from speed and extension patterns"""
        jab_threshold = np.mean(jab_speeds) + 1.5 * np.std(jab_speeds)
        potential_jabs = []
        
        i = 0
        while i < len(jab_speeds):
            if jab_speeds[i] > jab_threshold:
                # Found start of potential jab
                start = i
                
                # Find end of jab (when speed drops below threshold)
                while i < len(jab_speeds) and jab_speeds[i] > jab_threshold * 0.3:
                    i += 1
                
                end = i
                
                # Minimum jab duration check (to filter out noise)
                if end - start >= 3:  # At least 3 frames
                    jab = self._analyze_single_jab(normalized_features, jab_speeds, arm_angles, start, end)
                    if jab:
                        potential_jabs.append(jab)
            else:
                i += 1
                
        return potential_jabs
    
    def _analyze_single_jab(self, normalized_features, jab_speeds, arm_angles, start, end):
        """Analyze a single potential jab and return its metrics if valid"""
        # Calculate metrics for this jab
        jab_duration = end - start
        max_speed = max(jab_speeds[start:end]) if start < end else 0
        
        # Get arm extension (max angle during jab)
        angles_during_jab = arm_angles[start:end]
        max_extension = max(angles_during_jab) if angles_during_jab else None
        
        # Get normalized extension
        extensions = [features['r_extension'] for features in normalized_features[start:end]]
        max_norm_extension = max(extensions) if extensions else 0
        
        # Verify this is actually a jab by checking for extension pattern
        # A jab should show increasing extension followed by decreasing
        if len(extensions) >= 5:  # Need enough frames to detect pattern
            # Smooth the extension curve
            if len(extensions) >= 7:
                extensions = savgol_filter(extensions, min(7, len(extensions) - (len(extensions) % 2 - 1)), 2)
            
            # Check if extension increases then decreases
            max_idx = np.argmax(extensions)
            if max_idx > 1 and max_idx < len(extensions) - 2:
                # This looks like a jab pattern
                return {
                    'start': start,
                    'end': end,
                    'duration': jab_duration,
                    'max_speed': max_speed,
                    'max_extension': max_extension,
                    'max_norm_extension': max_norm_extension
                }
        else:
            # For short sequences, just use the speed
            return {
                'start': start,
                'end': end,
                'duration': jab_duration,
                'max_speed': max_speed,
                'max_extension': max_extension,
                'max_norm_extension': max_norm_extension
            }
        
        return None
    
    def _calculate_average_metrics(self, jabs):
        """Calculate average metrics across all jabs"""
        if not jabs:
            return {
                'avg_duration': 0,
                'avg_max_speed': 0,
                'avg_max_extension': None,
                'avg_norm_extension': 0
            }
            
        return {
            'avg_duration': np.mean([jab['duration'] for jab in jabs]),
            'avg_max_speed': np.mean([jab['max_speed'] for jab in jabs]),
            'avg_max_extension': np.mean([jab['max_extension'] for jab in jabs if jab['max_extension'] is not None]) 
                            if any(jab['max_extension'] is not None for jab in jabs) else None,
            'avg_norm_extension': np.mean([jab['max_norm_extension'] for jab in jabs])
        }
    
    def _compare_to_reference(self, pose_data, user_analysis, reference_name):
        """Compare user jabs to reference model"""
        logger.info(f"Comparing to reference model: {reference_name}")
        ref_data = self.references[reference_name]['poses']
        
        # Extract reference jabs
        ref_analysis = self.analyze_jab(ref_data)
        ref_jabs = ref_analysis['jabs']
        
        if not ref_jabs or not user_analysis['jabs']:
            logger.warning("No jabs available for comparison")
            return None
            
        # Compare each jab to reference jabs
        jab_similarities = self._compare_individual_jabs(
            pose_data, 
            user_analysis['jabs'], 
            ref_data, 
            ref_jabs
        )
        
        # Calculate overall metrics comparison
        metrics_comparison = self._calculate_metrics_comparison(
            user_analysis['avg_metrics'], 
            ref_analysis['avg_metrics']
        )
        
        comparison_result = {
            'similarities': jab_similarities,
            'average_similarity': np.mean([s['similarity'] for s in jab_similarities]) if jab_similarities else 0,
            'metrics_comparison': metrics_comparison
        }
        
        if comparison_result['average_similarity'] > 0:
            logger.info(f"Average similarity to reference: {comparison_result['average_similarity']*100:.1f}%")
        else:
            logger.info("Could not calculate precise similarity, but metrics comparison is available")
            
        return comparison_result
    
    def _compare_individual_jabs(self, user_data, user_jabs, ref_data, ref_jabs):
        """Compare each user jab to reference jabs and find best matches"""
        jab_similarities = []
        
        for user_jab in user_jabs:
            user_start, user_end = user_jab['start'], user_jab['end']
            user_jab_frames = user_data[user_start:user_end+1]
            
            best_similarity = 0
            best_ref_jab = None
            
            for ref_jab in ref_jabs:
                ref_start, ref_end = ref_jab['start'], ref_jab['end']
                ref_jab_frames = ref_data[ref_start:ref_end+1]
                
                # Use our improved orientation-invariant comparison
                similarity = self._compare_sequences(user_jab_frames, ref_jab_frames)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_ref_jab = ref_jab
            
            jab_similarities.append({
                'user_jab': user_jab,
                'ref_jab': best_ref_jab,
                'similarity': best_similarity
            })
            
        return jab_similarities
    
    def _calculate_metrics_comparison(self, user_metrics, ref_metrics):
        """Calculate comparison ratios between user and reference metrics"""
        return {
            'duration_ratio': user_metrics['avg_duration'] / ref_metrics['avg_duration'] 
                            if ref_metrics['avg_duration'] > 0 else 0,
            'speed_ratio': user_metrics['avg_max_speed'] / ref_metrics['avg_max_speed'] 
                         if ref_metrics['avg_max_speed'] > 0 else 0,
            'extension_ratio': user_metrics['avg_max_extension'] / ref_metrics['avg_max_extension'] 
                            if (user_metrics['avg_max_extension'] is not None and 
                                ref_metrics['avg_max_extension'] is not None and
                                ref_metrics['avg_max_extension'] > 0) else 0,
            'norm_extension_ratio': user_metrics['avg_norm_extension'] / ref_metrics['avg_norm_extension']
                                 if (user_metrics['avg_norm_extension'] > 0 and 
                                     ref_metrics['avg_norm_extension'] > 0) else 0
        }
    
    def _extract_normalized_features(self, sequence):
        """Extract normalized features that are invariant to video orientation"""
        features = []
        
        for frame in sequence:
            frame_features = {}
            
            # Check if required landmarks are present
            required_landmarks = ['right_shoulder', 'right_elbow', 'right_wrist', 
                                 'left_shoulder', 'left_elbow', 'left_wrist']
            
            if not all(lm in frame for lm in required_landmarks):
                continue
                
            # 1. Arm angles (orientation invariant)
            # Right arm angle
            r_shoulder = np.array([frame['right_shoulder']['x'], frame['right_shoulder']['y']])
            r_elbow = np.array([frame['right_elbow']['x'], frame['right_elbow']['y']])
            r_wrist = np.array([frame['right_wrist']['x'], frame['right_wrist']['y']])
            
            r_upper_arm = r_elbow - r_shoulder
            r_forearm = r_wrist - r_elbow
            
            # Calculate angle in degrees
            r_cos_angle = np.dot(r_upper_arm, r_forearm) / (
                np.linalg.norm(r_upper_arm) * np.linalg.norm(r_forearm)
            )
            r_angle = np.degrees(np.arccos(np.clip(r_cos_angle, -1.0, 1.0)))
            
            # Left arm angle
            l_shoulder = np.array([frame['left_shoulder']['x'], frame['left_shoulder']['y']])
            l_elbow = np.array([frame['left_elbow']['x'], frame['left_elbow']['y']])
            l_wrist = np.array([frame['left_wrist']['x'], frame['left_wrist']['y']])
            
            l_upper_arm = l_elbow - l_shoulder
            l_forearm = l_wrist - l_elbow
            
            l_cos_angle = np.dot(l_upper_arm, l_forearm) / (
                np.linalg.norm(l_upper_arm) * np.linalg.norm(l_forearm)
            )
            l_angle = np.degrees(np.arccos(np.clip(l_cos_angle, -1.0, 1.0)))
            
            # 2. Normalized arm extension (relative to body size)
            # Use shoulder width as a normalization factor
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
            
            r_extension = np.linalg.norm(r_wrist - r_shoulder) / shoulder_width if shoulder_width > 0 else 0
            l_extension = np.linalg.norm(l_wrist - l_shoulder) / shoulder_width if shoulder_width > 0 else 0
            
            # 3. Arm direction vectors (normalized)
            if np.linalg.norm(r_wrist - r_shoulder) > 0:
                r_direction = (r_wrist - r_shoulder) / np.linalg.norm(r_wrist - r_shoulder)
            else:
                r_direction = np.array([0, 0])
                
            if np.linalg.norm(l_wrist - l_shoulder) > 0:
                l_direction = (l_wrist - l_shoulder) / np.linalg.norm(l_wrist - l_shoulder)
            else:
                l_direction = np.array([0, 0])
            
            # Store features
            frame_features = {
                'r_angle': r_angle,
                'l_angle': l_angle,
                'r_extension': r_extension,
                'l_extension': l_extension,
                'r_direction_x': r_direction[0],
                'r_direction_y': r_direction[1],
                'l_direction_x': l_direction[0],
                'l_direction_y': l_direction[1]
            }
            
            features.append(frame_features)
        
        return features
    
    @timed
    def _compare_sequences(self, seq1, seq2):
        """
        Compare two pose sequences and return similarity score (0-1)
        Uses orientation-invariant comparison and dynamic time warping
        """
        if not seq1 or not seq2:
            logger.warning("Empty sequence provided for comparison")
            return 0
            
        # Extract key features for comparison
        seq1_features = self._extract_normalized_features(seq1)
        seq2_features = self._extract_normalized_features(seq2)
        
        if not seq1_features or not seq2_features:
            logger.warning("Could not extract features for comparison")
            return 0
        
        # Apply dynamic time warping for temporal alignment
        similarity = self._dtw_similarity(seq1_features, seq2_features)
        
        logger.debug(f"Sequence similarity: {similarity:.3f}")
        return similarity
    
    def _dtw_similarity(self, seq1, seq2, max_warping=0.5):
        """
        Calculate similarity between sequences using Dynamic Time Warping
        Returns a similarity score between 0 and 1
        """
        if not seq1 or not seq2:
            return 0
            
        # For simplicity, if sequences are very different in length, resample the longer one
        if len(seq1) > len(seq2) * (1 + max_warping) or len(seq2) > len(seq1) * (1 + max_warping):
            if len(seq1) > len(seq2):
                indices = np.linspace(0, len(seq1)-1, len(seq2)).astype(int)
                seq1 = [seq1[i] for i in indices]
            else:
                indices = np.linspace(0, len(seq2)-1, len(seq1)).astype(int)
                seq2 = [seq2[i] for i in indices]
        
        # Initialize DTW matrix
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, 1:] = np.inf
        dtw_matrix[1:, 0] = np.inf
        
        # Calculate DTW matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = self._feature_distance(seq1[i-1], seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        # Calculate similarity from DTW distance
        # Normalize by path length
        path_length = n + m
        normalized_distance = dtw_matrix[n, m] / path_length
        
        # Convert to similarity (0-1)
        # Use an exponential decay function to convert distance to similarity
        similarity = np.exp(-normalized_distance)
        
        return similarity
    
    def _feature_distance(self, features1, features2):
        """Calculate distance between two feature sets"""
        # Weight each feature type
        weights = {
            'r_angle': 0.3,       # Right arm angle
            'l_angle': 0.1,       # Left arm angle (less important for jab)
            'r_extension': 0.3,   # Right arm extension
            'l_extension': 0.1,   # Left arm extension
            'r_direction_x': 0.1, # Right arm direction X
            'r_direction_y': 0.1, # Right arm direction Y
            'l_direction_x': 0.0, # Left arm direction X (not used)
            'l_direction_y': 0.0  # Left arm direction Y (not used)
        }
        
        total_distance = 0
        total_weight = 0
        
        for feature, weight in weights.items():
            if feature in features1 and feature in features2:
                # For angles, use circular distance
                if 'angle' in feature:
                    angle1 = features1[feature]
                    angle2 = features2[feature]
                    # Circular distance for angles
                    angle_dist = min(abs(angle1 - angle2), 360 - abs(angle1 - angle2)) / 180.0
                    total_distance += weight * angle_dist
                else:
                    # Euclidean distance for other features
                    total_distance += weight * abs(features1[feature] - features2[feature])
                
                total_weight += weight
        
        if total_weight > 0:
            return total_distance / total_weight
        else:
            return float('inf')
    
    @timed
    def visualize_comparison(self, user_data, reference_name, output_path):
        """Create a visualization comparing user technique to reference"""
        logger.info(f"Creating visualization comparing user technique to reference '{reference_name}'")
        
        if reference_name not in self.references:
            logger.error(f"Reference '{reference_name}' not found")
            return
            
        reference_data = self.references[reference_name]['poses']
        
        # Analyze both
        logger.info("Analyzing user technique...")
        user_analysis = self.analyze_jab(user_data)
        
        logger.info("Analyzing reference technique...")
        ref_analysis = self.analyze_jab(reference_data)
        
        # Create visualization
        self._create_comparison_plots(user_analysis, ref_analysis, reference_data, user_data, output_path)
        
        # Generate feedback and metrics summary
        feedback = self._generate_feedback(user_analysis, ref_analysis)
        metrics_summary = self._generate_metrics_summary(user_analysis, ref_analysis)
        
        # Return analysis results
        analysis_results = {
            'user_jabs_detected': len(user_analysis['jabs']),
            'reference_jabs_detected': len(ref_analysis['jabs']),
            'metrics': metrics_summary,
            'feedback': feedback
        }
        
        # Add overall score if available
        if 'overall_score' in metrics_summary:
            analysis_results['overall_score'] = f"{metrics_summary['overall_score']:.1f}%"
            logger.info(f"Analysis complete: {len(user_analysis['jabs'])} jabs detected, "
                       f"overall score: {metrics_summary['overall_score']:.1f}%")
        elif 'similarity_score' in metrics_summary:
            analysis_results['similarity_score'] = f"{metrics_summary['similarity_score']:.1f}%"
            logger.info(f"Analysis complete: {len(user_analysis['jabs'])} jabs detected, "
                       f"similarity score: {metrics_summary['similarity_score']:.1f}%")
        else:
            logger.info(f"Analysis complete: {len(user_analysis['jabs'])} jabs detected, metrics comparison available")
            
        return analysis_results
    
    def _create_comparison_plots(self, user_analysis, ref_analysis, reference_data, user_data, output_path):
        """Create comparison visualization plots"""
        logger.info(f"Generating comparison visualization to {output_path}")
        plt.figure(figsize=(15, 15))  # Larger figure for more plots
        
        # 1. Arm Angle Comparison
        self._plot_arm_angle_comparison(user_analysis, ref_analysis, reference_data, user_data)
        
        # 2. Jab Speed Comparison
        self._plot_jab_speed_comparison(user_analysis)
        
        # 3. Metrics Comparison
        self._plot_metrics_comparison(user_analysis, ref_analysis)
        
        # Save the figure
        plt.tight_layout()
        Path(output_path).parent.mkdir(exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def _plot_arm_angle_comparison(self, user_analysis, ref_analysis, reference_data, user_data):
        """Plot arm angle comparison between user and reference"""
        plt.subplot(3, 1, 1)
        plt.title('Arm Angle Comparison')
        plt.plot(user_analysis['arm_angles'], label='You', color='blue')
        
        # Highlight detected jabs
        for jab in user_analysis['jabs']:
            plt.axvspan(jab['start'], jab['end'], color='blue', alpha=0.3)
            # Add a label for each jab
            mid_point = (jab['start'] + jab['end']) // 2
            if mid_point < len(user_analysis['arm_angles']):
                plt.text(mid_point, user_analysis['arm_angles'][mid_point] + 10, 
                        f"Jab {user_analysis['jabs'].index(jab)+1}", 
                        color='blue', fontsize=8)
        
        # Plot reference on same time scale if available
        if reference_data:
            # Resample reference to match user data length
            if len(reference_data) != len(user_data):
                indices = np.linspace(0, len(ref_analysis['arm_angles'])-1, len(user_analysis['arm_angles'])).astype(int)
                resampled_angles = [ref_analysis['arm_angles'][i] for i in indices]
            else:
                resampled_angles = ref_analysis['arm_angles']
                
            plt.plot(resampled_angles, label='Reference', color='green')
            
            # Highlight reference jabs
            for jab in ref_analysis['jabs']:
                # Scale jab positions to match user data timeline
                scaled_start = int(jab['start'] * len(user_data) / len(reference_data))
                scaled_end = int(jab['end'] * len(user_data) / len(reference_data))
                plt.axvspan(scaled_start, scaled_end, color='green', alpha=0.3)
        
        plt.ylabel('Arm Angle (degrees)')
        plt.legend()
    
    def _plot_jab_speed_comparison(self, user_analysis):
        """Plot jab speed comparison"""
        plt.subplot(3, 1, 2)
        plt.title('Jab Speed Comparison')
        plt.plot(user_analysis['jab_speeds'], label='You', color='blue')
        
        # Highlight detected jabs
        for jab in user_analysis['jabs']:
            plt.axvspan(jab['start'], jab['end'], color='blue', alpha=0.3)
        
        # Plot threshold
        threshold = np.mean(user_analysis['jab_speeds']) + 1.5 * np.std(user_analysis['jab_speeds'])
        plt.axhline(y=threshold, color='red', linestyle='--', label='Jab Threshold')
        
        plt.ylabel('Speed (normalized)')
        plt.xlabel('Frame')
        plt.legend()
    
    def _plot_metrics_comparison(self, user_analysis, ref_analysis):
        """Plot metrics comparison between user and reference"""
        plt.subplot(3, 1, 3)
        plt.title('Technique Metrics Comparison')
        
        # Prepare metrics for bar chart
        metrics = ['Duration', 'Speed', 'Extension']
        user_values = [
            user_analysis['avg_metrics']['avg_duration'],
            user_analysis['avg_metrics']['avg_max_speed'],
            user_analysis['avg_metrics']['avg_max_extension'] or 0
        ]
        
        ref_values = [
            ref_analysis['avg_metrics']['avg_duration'],
            ref_analysis['avg_metrics']['avg_max_speed'],
            ref_analysis['avg_metrics']['avg_max_extension'] or 0
        ]
        
        # Normalize values for better visualization
        max_values = [max(u, r) for u, r in zip(user_values, ref_values)]
        user_norm = [u/m if m > 0 else 0 for u, m in zip(user_values, max_values)]
        ref_norm = [r/m if m > 0 else 0 for r, m in zip(ref_values, max_values)]
        
        # Create bar chart
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, user_norm, width, label='You', color='blue')
        plt.bar(x + width/2, ref_norm, width, label='Reference', color='green')
        
        # Add value labels
        for i, v in enumerate(user_values):
            plt.text(i - width/2, user_norm[i] + 0.05, f"{v:.1f}", ha='center', va='bottom', color='blue', fontsize=8)
        
        for i, v in enumerate(ref_values):
            plt.text(i + width/2, ref_norm[i] + 0.05, f"{v:.1f}", ha='center', va='bottom', color='green', fontsize=8)
        
        plt.xticks(x, metrics)
        plt.ylabel('Normalized Value')
        plt.ylim(0, 1.3)  # Leave room for labels
        plt.legend()
    
    def _generate_metrics_summary(self, user_analysis, ref_analysis):
        """Generate metrics summary comparing user to reference"""
        metrics_summary = {}
        
        # Basic metrics
        metrics_summary['jabs_detected'] = {
            'user': len(user_analysis['jabs']),
            'reference': len(ref_analysis['jabs'])
        }
        
        # Add average metrics
        metrics_summary['avg_jab_duration'] = {
            'user': user_analysis['avg_metrics']['avg_duration'],
            'reference': ref_analysis['avg_metrics']['avg_duration']
        }
        
        metrics_summary['avg_jab_speed'] = {
            'user': user_analysis['avg_metrics']['avg_max_speed'],
            'reference': ref_analysis['avg_metrics']['avg_max_speed']
        }
        
        metrics_summary['avg_arm_extension'] = {
            'user': user_analysis['avg_metrics']['avg_max_extension'],
            'reference': ref_analysis['avg_metrics']['avg_max_extension']
        }
        
        # Add comparison metrics if available
        if user_analysis['reference_comparison'] and 'metrics_comparison' in user_analysis['reference_comparison']:
            metrics_summary['comparison'] = user_analysis['reference_comparison']['metrics_comparison']
            
            # Calculate overall technique score if possible
            metrics_summary.update(self._calculate_overall_score(user_analysis['reference_comparison']))
        
        # Add similarity score if available
        if user_analysis['reference_comparison'] and user_analysis['reference_comparison']['average_similarity'] > 0:
            metrics_summary['similarity_score'] = user_analysis['reference_comparison']['average_similarity'] * 100
        
        return metrics_summary
    
    def _calculate_overall_score(self, comparison):
        """Calculate overall technique score based on comparison metrics"""
        result = {}
        
        if 'metrics_comparison' not in comparison:
            return result
            
        comp = comparison['metrics_comparison']
        
        # Weight the factors: speed (30%), extension (30%), normalized extension (30%), duration (10%)
        if all(v > 0 for v in [comp['speed_ratio'], comp['extension_ratio'], comp['norm_extension_ratio']]):
            # Normalize each ratio to be centered around 1.0 (where 1.0 means identical to reference)
            # Cap at 0.5 to 1.5 range (50% to 150% of reference)
            duration_score = min(max(comp['duration_ratio'], 0.5), 1.5)
            speed_score = min(max(comp['speed_ratio'], 0.5), 1.5)
            extension_score = min(max(comp['extension_ratio'], 0.5), 1.5)
            norm_extension_score = min(max(comp['norm_extension_ratio'], 0.5), 1.5)
            
            # Convert to percentage where 1.0 (perfect match) = 100%
            # and deviations in either direction reduce the score
            duration_percent = (1 - abs(duration_score - 1)) * 100
            speed_percent = (1 - abs(speed_score - 1)) * 100
            extension_percent = (1 - abs(extension_score - 1)) * 100
            norm_extension_percent = (1 - abs(norm_extension_score - 1)) * 100
            
            # Weighted average
            overall_score = (
                0.1 * duration_percent + 
                0.3 * speed_percent + 
                0.3 * extension_percent +
                0.3 * norm_extension_percent
            )
            
            result['overall_score'] = overall_score
            
        return result
    
    @timed
    def _generate_feedback(self, user_analysis, ref_analysis):
        """Generate detailed feedback based on comparison to reference"""
        feedback = []
        
        # Check if we have valid data to generate feedback
        if not self._has_valid_jab_data(user_analysis, ref_analysis, feedback):
            return feedback
            
        # Get comparison data
        comparison = user_analysis.get('reference_comparison', None)
        metrics_comp = self._get_metrics_comparison(comparison, feedback)
        if not metrics_comp:
            return feedback
            
        # Generate feedback on different aspects
        self._add_jab_count_feedback(user_analysis, ref_analysis, feedback)
        self._add_duration_feedback(metrics_comp, feedback)
        self._add_speed_feedback(metrics_comp, feedback)
        self._add_extension_feedback(metrics_comp, feedback)
        self._add_similarity_feedback(comparison, feedback)
        self._add_specific_jab_feedback(comparison, user_analysis, feedback)
        self._add_summary_feedback(metrics_comp, feedback)
        
        return feedback
    
    def _has_valid_jab_data(self, user_analysis, ref_analysis, feedback):
        """Check if we have valid jab data to generate feedback"""
        if not user_analysis['jabs']:
            feedback.append("No jabs detected in your video. Try performing clearer jab movements.")
            return False
            
        if not ref_analysis['jabs']:
            feedback.append("No jabs detected in reference video. Please select a different reference.")
            return False
            
        return True
    
    def _get_metrics_comparison(self, comparison, feedback):
        """Get metrics comparison data if available"""
        if not comparison:
            feedback.append("Could not compare to reference. Try recording with better lighting and camera angle.")
            return None
            
        metrics_comp = comparison.get('metrics_comparison', None)
        if not metrics_comp:
            feedback.append("Could not calculate comparison metrics. Try recording with better lighting and camera angle.")
            return None
            
        return metrics_comp
    
    def _add_jab_count_feedback(self, user_analysis, ref_analysis, feedback):
        """Add feedback about jab count"""
        user_jab_count = len(user_analysis['jabs'])
        ref_jab_count = len(ref_analysis['jabs'])
        
        if user_jab_count < ref_jab_count:
            feedback.append(f"You performed {user_jab_count} jabs, while the reference shows {ref_jab_count}. "
                           f"Try to complete the full sequence.")
        elif user_jab_count > ref_jab_count:
            feedback.append(f"You performed {user_jab_count} jabs, while the reference shows {ref_jab_count}. "
                           f"Focus on quality over quantity.")
        else:
            feedback.append(f"Good job matching the reference with {user_jab_count} jabs.")
    
    def _add_duration_feedback(self, metrics_comp, feedback):
        """Add feedback about jab duration"""
        duration_ratio = metrics_comp.get('duration_ratio', 0)
        if duration_ratio > 0:
            if duration_ratio < 0.8:
                feedback.append("Your jabs are faster than the reference. This can be good for speed, "
                               "but make sure you're maintaining proper form.")
            elif duration_ratio > 1.2:
                feedback.append("Your jabs are slower than the reference. Try to increase your speed "
                               "while maintaining proper form.")
            else:
                feedback.append("Good job matching the reference jab speed.")
    
    def _add_speed_feedback(self, metrics_comp, feedback):
        """Add feedback about jab speed"""
        speed_ratio = metrics_comp.get('speed_ratio', 0)
        if speed_ratio > 0:
            if speed_ratio < 0.8:
                feedback.append("Your jab speed is lower than the reference. Try to generate more power "
                               "from your hips and shoulders.")
            elif speed_ratio > 1.2:
                feedback.append("Your jab speed is higher than the reference. Great power generation, "
                               "but ensure you maintain control and accuracy.")
            else:
                feedback.append("Good job matching the reference jab power and speed.")
    
    def _add_extension_feedback(self, metrics_comp, feedback):
        """Add feedback about arm extension"""
        extension_ratio = metrics_comp.get('extension_ratio', 0)
        norm_extension_ratio = metrics_comp.get('norm_extension_ratio', 0)
        
        # Use normalized extension if available, otherwise use raw extension
        if norm_extension_ratio > 0:
            self._add_extension_ratio_feedback(norm_extension_ratio, feedback)
        elif extension_ratio > 0:
            self._add_extension_ratio_feedback(extension_ratio, feedback)
    
    def _add_extension_ratio_feedback(self, ratio, feedback):
        """Add feedback based on extension ratio"""
        if ratio < 0.8:
            feedback.append("You're not fully extending your arm during jabs. Try to extend more "
                           "while maintaining proper form and guard position.")
        elif ratio > 1.2:
            feedback.append("You may be over-extending your jabs. Be careful not to hyperextend "
                           "your elbow or compromise your guard.")
        else:
            feedback.append("Good job with your jab extension. You're reaching the optimal distance.")
    
    def _add_similarity_feedback(self, comparison, feedback):
        """Add feedback about overall similarity"""
        similarity = comparison.get('average_similarity', 0) * 100
        if similarity > 0:
            if similarity < 60:
                feedback.append(f"Your overall technique similarity is {similarity:.1f}%. There's significant room "
                               f"for improvement in matching the reference technique.")
            elif similarity < 80:
                feedback.append(f"Your overall technique similarity is {similarity:.1f}%. You're on the right track, "
                               f"but continue refining your technique to better match the reference.")
            else:
                feedback.append(f"Excellent job! Your overall technique similarity is {similarity:.1f}%. "
                               f"You're closely matching the reference technique.")
    
    def _add_specific_jab_feedback(self, comparison, user_analysis, feedback):
        """Add feedback about specific jabs"""
        if 'similarities' not in comparison or not comparison['similarities']:
            return
            
        # Find the best and worst jabs
        jab_similarities = comparison['similarities']
        jab_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        best_jab = jab_similarities[0]
        worst_jab = jab_similarities[-1]
        
        # Give feedback on best jab
        best_jab_idx = user_analysis['jabs'].index(best_jab['user_jab']) + 1
        feedback.append(f"Your best jab was #{best_jab_idx} with {best_jab['similarity']*100:.1f}% similarity "
                       f"to the reference. Try to replicate this technique consistently.")
        
        # Give feedback on worst jab if it's significantly worse
        if len(jab_similarities) > 1 and (best_jab['similarity'] - worst_jab['similarity']) > 0.2:
            worst_jab_idx = user_analysis['jabs'].index(worst_jab['user_jab']) + 1
            feedback.append(f"Your jab #{worst_jab_idx} had the lowest similarity at {worst_jab['similarity']*100:.1f}%. "
                           f"Focus on improving consistency across all jabs.")
    
    def _add_summary_feedback(self, metrics_comp, feedback):
        """Add summary feedback with overall score"""
        if 'overall_score' not in metrics_comp:
            return
            
        score = metrics_comp['overall_score']
        if score < 60:
            feedback.append(f"Overall score: {score:.1f}%. Focus on improving your jab technique by watching "
                           f"the reference video carefully and practicing the correct form.")
        elif score < 80:
            feedback.append(f"Overall score: {score:.1f}%. You're showing good progress. Continue practicing "
                           f"to refine your technique and increase consistency.")
        else:
            feedback.append(f"Overall score: {score:.1f}%. Excellent work! You're demonstrating great technique. "
                           f"Keep practicing to maintain this high level of performance.")
    
# Example usage
if __name__ == "__main__":
    logger.info("=== ShadowCoach Boxing Analysis System ===")
    logger.info("Starting analysis process...")
    
    start_time = time.time()
    analyzer = BoxingAnalyzer()
    
    # Define video paths
    reference_video = "assets/How to Throw a Jab.mp4"
    user_video_path = "assets/The Wrong Jab.mp4"
    
    # Create output paths
    reference_output = generate_unique_filename("output", "reference_analyzed", ".mp4")
    user_output = generate_unique_filename("output", "user_analyzed", ".mp4")
    comparison_output = generate_unique_filename("output", "jab_comparison", ".png")
    
    # Create output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    logger.info(f"Processing reference video: {reference_video}")
    # Process reference video
    ref_poses = analyzer.process_video(
        reference_video, 
        output_path=reference_output
    )
    
    # Save reference
    analyzer.save_reference("pro_boxer", ref_poses, {
        "boxer_name": "Professional Example",
        "style": "Orthodox",
        "technique": "Jab practice"
    })
    
    # Check if user video exists
    user_video_exists = Path(user_video_path).exists()
    
    if user_video_exists:
        logger.info(f"Processing user video: {user_video_path}")
        # Process user video
        user_poses = analyzer.process_video(
            user_video_path, 
            output_path=user_output
        )
        
        # Analyze and compare
        logger.info("Comparing user technique to reference...")
        comparison = analyzer.visualize_comparison(
            user_poses, 
            "pro_boxer",
            comparison_output
        )
        
        logger.info("\nAnalysis Results:")
        for key, value in comparison.items():
            logger.info(f"{key}: {value}")
            
            if isinstance(value, list):
                for item in value:
                    logger.info(f"  - {item}")
    else:
        logger.warning(f"User video not found: {user_video_path}")
        logger.info("To analyze your technique, record a video of yourself throwing jabs")
        logger.info("and save it as 'assets/user_jab.mp4'")
        
    # Save the reference model for future use
    model_path = generate_unique_filename("models", "jab_reference", ".pkl")
    analyzer.save_references_to_file(model_path)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Print detailed timing summary
    logger.info("\n=== Performance Summary ===")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    # Print timing for each function
    logger.info("\nDetailed Function Timing:")
    for func_name, times in function_timings.items():
        # Calculate statistics
        count = len(times)
        total = sum(times)
        avg = total / count if count > 0 else 0
        max_time = max(times) if times else 0
        min_time = min(times) if times else 0
        
        # Print detailed timing information
        logger.info(f"  {func_name}:")
        logger.info(f"    Calls: {count}")
        logger.info(f"    Total time: {total:.2f}s")
        logger.info(f"    Average time: {avg:.2f}s")
        logger.info(f"    Min/Max: {min_time:.2f}s / {max_time:.2f}s")
        logger.info(f"    Percentage of total: {(total/total_time)*100:.1f}%")
    
    # Print output file information
    logger.info("\n=== Output Files ===")
    logger.info(f"Reference video analysis: {reference_output}")
    if user_video_exists:
        logger.info(f"User video analysis: {user_output}")
        logger.info(f"Technique comparison: {comparison_output}")
    logger.info(f"Reference model: {model_path}")
    
    logger.info("\nProcessing complete! You can now review the analysis results in the output folder.")