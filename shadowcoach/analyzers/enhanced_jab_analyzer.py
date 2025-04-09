"""
Enhanced jab technique analyzer with advanced boxing fundamentals
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.signal import savgol_filter

from ..utils.logging_utils import logger, timed
from ..core.reference_manager import ReferenceManager
from .jab_analyzer import JabAnalyzer

class EnhancedJabAnalyzer(JabAnalyzer):
    """
    Enhanced class for analyzing jab boxing techniques with detailed fundamentals
    """
    def __init__(self, reference_manager: ReferenceManager):
        """
        Initialize the enhanced jab analyzer
        
        Args:
            reference_manager: Reference manager for comparing techniques
        """
        super().__init__(reference_manager)
        logger.info("Initializing EnhancedJabAnalyzer...")
        logger.info("EnhancedJabAnalyzer initialized successfully")
    
    @timed
    def analyze_jab(self, pose_data: List[Dict[str, Dict[str, float]]], 
                   reference_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze jab technique by comparing to reference with enhanced fundamentals
        
        Args:
            pose_data: The pose data to analyze
            reference_name: Optional reference name to compare against
            
        Returns:
            Analysis metrics with enhanced fundamentals
        """
        # Get basic analysis from parent class
        basic_analysis = super().analyze_jab(pose_data, reference_name)
        
        # If no jabs detected, return basic analysis
        if not basic_analysis['jabs']:
            return basic_analysis
        
        # Extract enhanced features for each jab
        enhanced_features = self._extract_enhanced_features(pose_data, basic_analysis['jabs'])
        
        # Add enhanced features to the analysis
        basic_analysis['enhanced_features'] = enhanced_features
        
        # Generate technical scores
        technical_scores = self._evaluate_technical_elements(enhanced_features)
        basic_analysis['technical_scores'] = technical_scores
        
        return basic_analysis
    
    def get_reference_analysis(self, reference_name: str) -> Dict[str, Any]:
        """
        Get analysis of a reference technique
        
        Args:
            reference_name: Name of the reference to analyze
            
        Returns:
            Analysis metrics for the reference
        """
        logger.info(f"Getting analysis for reference: {reference_name}")
        
        # Get reference data
        reference = self.reference_manager.get_reference(reference_name)
        if not reference:
            logger.error(f"Reference '{reference_name}' not found")
            return {
                'arm_angles': [],
                'jab_speeds': [],
                'jabs': [],
                'avg_metrics': {
                    'avg_duration': 0,
                    'avg_max_speed': 0,
                    'avg_max_extension': None
                }
            }
        
        # Analyze reference jabs
        ref_analysis = self.analyze_jab(reference['poses'])
        
        return ref_analysis
    
    def _extract_enhanced_features(self, pose_data: List[Dict[str, Dict[str, float]]], 
                                 jabs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract enhanced features for each detected jab
        
        Args:
            pose_data: The pose data
            jabs: List of detected jabs
            
        Returns:
            Enhanced features for each jab
        """
        enhanced_features = []
        
        for jab in jabs:
            start_frame = jab['start']
            end_frame = jab['end']
            
            # Ensure we have enough frames
            if start_frame >= len(pose_data) or end_frame >= len(pose_data):
                continue
                
            # Get frames for this jab
            jab_frames = pose_data[start_frame:end_frame+1]
            
            # Extract individual analyses first to avoid redundant calculations
            starting_position = self._analyze_starting_position(jab_frames)
            extension_trajectory = self._analyze_extension_trajectory(jab_frames)
            hand_position = self._analyze_hand_position(jab_frames)
            body_mechanics = self._analyze_body_mechanics(jab_frames)
            retraction = self._analyze_retraction(jab_frames)
            
            # Detect common errors using the analyses we've already performed
            common_errors = self._detect_common_errors(jab_frames, extension_trajectory, retraction)
            
            # Compile features
            features = {
                'starting_position': starting_position,
                'extension_trajectory': extension_trajectory,
                'hand_position': hand_position,
                'body_mechanics': body_mechanics,
                'retraction': retraction,
                'common_errors': common_errors
            }
            
            enhanced_features.append(features)
            
        return enhanced_features
    
    def _analyze_starting_position(self, frames: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Analyze the starting position of the jab
        
        Args:
            frames: Frames of the jab
            
        Returns:
            Analysis of starting position
        """
        # Get the first few frames
        start_frames = frames[:min(5, len(frames))]
        
        # Check if we have all required landmarks
        if not start_frames or not all(
            all(landmark in frame for landmark in ['right_shoulder', 'right_elbow', 'right_wrist', 'nose'])
            for frame in start_frames
        ):
            return {'quality': 0, 'issues': ['Missing landmarks for starting position analysis']}
        
        # Extract first frame for starting position
        first_frame = start_frames[0]
        
        # Check hand position relative to face (guard position)
        nose = np.array([first_frame['nose']['x'], first_frame['nose']['y']])
        wrist = np.array([first_frame['right_wrist']['x'], first_frame['right_wrist']['y']])
        shoulder = np.array([first_frame['right_shoulder']['x'], first_frame['right_shoulder']['y']])
        
        # Distance from wrist to nose (should be small in guard position)
        face_distance = np.linalg.norm(wrist - nose)
        
        # Normalize by shoulder width
        if 'left_shoulder' in first_frame:
            left_shoulder = np.array([first_frame['left_shoulder']['x'], first_frame['left_shoulder']['y']])
            shoulder_width = np.linalg.norm(shoulder - left_shoulder)
            normalized_face_distance = face_distance / shoulder_width
        else:
            normalized_face_distance = face_distance
        
        # Check elbow position (should be down and tucked)
        elbow = np.array([first_frame['right_elbow']['x'], first_frame['right_elbow']['y']])
        elbow_height = elbow[1] - shoulder[1]  # Positive if elbow is below shoulder
        
        # Evaluate starting position
        issues = []
        quality = 1.0  # Start with perfect score
        
        # Check hand position
        if normalized_face_distance > 0.3:
            issues.append('Hand not in proper guard position (too far from face)')
            quality -= 0.3
        
        # Check elbow position
        if elbow_height < 0:
            issues.append('Elbow raised too high (not tucked)')
            quality -= 0.3
        
        return {
            'quality': max(0, quality),
            'guard_position_score': max(0, 1.0 - normalized_face_distance),
            'elbow_position_score': max(0, 1.0 - abs(elbow_height) if elbow_height >= 0 else 0.7),
            'issues': issues
        }
    
    def _analyze_extension_trajectory(self, frames: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Analyze the extension and trajectory of the jab
        
        Args:
            frames: Frames of the jab
            
        Returns:
            Analysis of extension and trajectory
        """
        if not frames or len(frames) < 3:
            return {'quality': 0, 'issues': ['Not enough frames for trajectory analysis']}
        
        # Get wrist positions throughout the jab
        wrist_positions = []
        for frame in frames:
            if 'right_wrist' not in frame:
                continue
            wrist_positions.append(np.array([
                frame['right_wrist']['x'],
                frame['right_wrist']['y'],
                frame['right_wrist']['z']
            ]))
        
        if len(wrist_positions) < 3:
            return {'quality': 0, 'issues': ['Missing wrist positions for trajectory analysis']}
        
        # Calculate trajectory straightness
        # A straight line will have minimal deviation from the start-to-end vector
        start_pos = wrist_positions[0]
        end_pos = wrist_positions[-1]
        direct_vector = end_pos - start_pos
        direct_distance = np.linalg.norm(direct_vector)
        
        # Calculate the actual path length
        path_length = 0
        for i in range(1, len(wrist_positions)):
            path_length += np.linalg.norm(wrist_positions[i] - wrist_positions[i-1])
        
        # Straightness ratio (1.0 = perfectly straight, higher values = more curved)
        straightness_ratio = path_length / direct_distance if direct_distance > 0 else float('inf')
        
        # Check for full extension
        # Get shoulder and elbow positions at maximum extension
        max_extension_idx = len(frames) // 2  # Approximate middle of the jab
        
        if 'right_shoulder' not in frames[max_extension_idx] or 'right_elbow' not in frames[max_extension_idx]:
            return {'quality': 0, 'issues': ['Missing landmarks for extension analysis']}
        
        shoulder = np.array([
            frames[max_extension_idx]['right_shoulder']['x'],
            frames[max_extension_idx]['right_shoulder']['y'],
            frames[max_extension_idx]['right_shoulder']['z']
        ])
        
        elbow = np.array([
            frames[max_extension_idx]['right_elbow']['x'],
            frames[max_extension_idx]['right_elbow']['y'],
            frames[max_extension_idx]['right_elbow']['z']
        ])
        
        wrist = np.array([
            frames[max_extension_idx]['right_wrist']['x'],
            frames[max_extension_idx]['right_wrist']['y'],
            frames[max_extension_idx]['right_wrist']['z']
        ])
        
        # Calculate arm angle at maximum extension
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        
        cos_angle = np.dot(upper_arm, forearm) / (
            np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
        )
        extension_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # Evaluate extension and trajectory
        issues = []
        quality = 1.0  # Start with perfect score
        
        # Check straightness (1.0-1.2 is good, higher values indicate looping)
        straightness_score = max(0, 1.0 - (straightness_ratio - 1.0))
        if straightness_ratio > 1.2:
            issues.append(f'Jab trajectory not straight (looping or arcing)')
            quality -= 0.3
        
        # Check extension angle (should be close to 180 degrees for full extension)
        extension_score = extension_angle / 180.0
        if extension_angle < 160:
            issues.append('Incomplete extension (not fully committing to the jab)')
            quality -= 0.3
        elif extension_angle > 190:  # Allow some margin for error in measurement
            issues.append('Hyperextending the elbow (overextending)')
            quality -= 0.2
        
        return {
            'quality': max(0, quality),
            'straightness_score': straightness_score,
            'extension_score': extension_score,
            'straightness_ratio': straightness_ratio,
            'extension_angle': extension_angle,
            'issues': issues
        }
    
    def _analyze_hand_position(self, frames: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Analyze the hand position during the jab
        
        Args:
            frames: Frames of the jab
            
        Returns:
            Analysis of hand position
        """
        # This is more challenging with just pose data
        # We would need hand orientation which isn't fully captured
        # We'll make some approximations based on available data
        
        if not frames or len(frames) < 3:
            return {'quality': 0, 'issues': ['Not enough frames for hand position analysis']}
        
        # For now, we'll return a placeholder with medium quality
        # In a real system, this would need hand keypoints or additional sensors
        return {
            'quality': 0.7,
            'issues': ['Hand rotation and knuckle alignment cannot be fully assessed with current data']
        }
    
    def _analyze_body_mechanics(self, frames: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Analyze the body mechanics during the jab
        
        Args:
            frames: Frames of the jab
            
        Returns:
            Analysis of body mechanics
        """
        if not frames or len(frames) < 3:
            return {'quality': 0, 'issues': ['Not enough frames for body mechanics analysis']}
        
        # Check if we have all required landmarks
        required_landmarks = ['right_shoulder', 'left_shoulder', 'right_hip', 'left_hip']
        if not all(all(landmark in frame for landmark in required_landmarks) for frame in frames):
            return {'quality': 0, 'issues': ['Missing landmarks for body mechanics analysis']}
        
        # Get first and middle frames for comparison
        first_frame = frames[0]
        mid_frame = frames[len(frames) // 2]
        
        # Check shoulder rotation
        first_shoulders = np.array([
            first_frame['right_shoulder']['x'] - first_frame['left_shoulder']['x'],
            first_frame['right_shoulder']['y'] - first_frame['left_shoulder']['y']
        ])
        
        mid_shoulders = np.array([
            mid_frame['right_shoulder']['x'] - mid_frame['left_shoulder']['x'],
            mid_frame['right_shoulder']['y'] - mid_frame['left_shoulder']['y']
        ])
        
        # Normalize
        first_shoulders = first_shoulders / np.linalg.norm(first_shoulders)
        mid_shoulders = mid_shoulders / np.linalg.norm(mid_shoulders)
        
        # Calculate angle change in shoulders
        shoulder_rotation = np.degrees(np.arccos(np.clip(np.dot(first_shoulders, mid_shoulders), -1.0, 1.0)))
        
        # Check hip rotation
        first_hips = np.array([
            first_frame['right_hip']['x'] - first_frame['left_hip']['x'],
            first_frame['right_hip']['y'] - first_frame['left_hip']['y']
        ])
        
        mid_hips = np.array([
            mid_frame['right_hip']['x'] - mid_frame['left_hip']['x'],
            mid_frame['right_hip']['y'] - mid_frame['left_hip']['y']
        ])
        
        # Normalize
        first_hips = first_hips / np.linalg.norm(first_hips)
        mid_hips = mid_hips / np.linalg.norm(mid_hips)
        
        # Calculate angle change in hips
        hip_rotation = np.degrees(np.arccos(np.clip(np.dot(first_hips, mid_hips), -1.0, 1.0)))
        
        # Check posture (vertical alignment)
        first_spine = np.array([
            (first_frame['right_shoulder']['x'] + first_frame['left_shoulder']['x']) / 2 - 
            (first_frame['right_hip']['x'] + first_frame['left_hip']['x']) / 2,
            (first_frame['right_shoulder']['y'] + first_frame['left_shoulder']['y']) / 2 - 
            (first_frame['right_hip']['y'] + first_frame['left_hip']['y']) / 2
        ])
        
        mid_spine = np.array([
            (mid_frame['right_shoulder']['x'] + mid_frame['left_shoulder']['x']) / 2 - 
            (mid_frame['right_hip']['x'] + mid_frame['left_hip']['x']) / 2,
            (mid_frame['right_shoulder']['y'] + mid_frame['left_shoulder']['y']) / 2 - 
            (mid_frame['right_hip']['y'] + mid_frame['left_hip']['y']) / 2
        ])
        
        # Normalize
        first_spine = first_spine / np.linalg.norm(first_spine)
        mid_spine = mid_spine / np.linalg.norm(mid_spine)
        
        # Calculate posture change
        posture_change = np.degrees(np.arccos(np.clip(np.dot(first_spine, mid_spine), -1.0, 1.0)))
        
        # Evaluate body mechanics
        issues = []
        quality = 1.0  # Start with perfect score
        
        # Check shoulder rotation (should be 5-15 degrees for a good jab)
        shoulder_score = min(1.0, shoulder_rotation / 15.0) if shoulder_rotation <= 15 else max(0, 1.0 - (shoulder_rotation - 15) / 15)
        if shoulder_rotation < 3:
            issues.append('Insufficient shoulder rotation (not protecting chin)')
            quality -= 0.2
        elif shoulder_rotation > 20:
            issues.append('Excessive shoulder rotation (overcommitting)')
            quality -= 0.2
        
        # Check hip rotation (should be slight, 3-10 degrees)
        hip_score = min(1.0, hip_rotation / 10.0) if hip_rotation <= 10 else max(0, 1.0 - (hip_rotation - 10) / 10)
        if hip_rotation < 2:
            issues.append('Insufficient hip rotation (not generating power)')
            quality -= 0.2
        elif hip_rotation > 15:
            issues.append('Excessive hip rotation (overcommitting)')
            quality -= 0.2
        
        # Check posture change (should be minimal, <5 degrees)
        posture_score = max(0, 1.0 - posture_change / 10.0)
        if posture_change > 10:
            issues.append('Poor posture maintenance (leaning or dropping)')
            quality -= 0.3
        
        return {
            'quality': max(0, quality),
            'shoulder_rotation_score': shoulder_score,
            'hip_rotation_score': hip_score,
            'posture_score': posture_score,
            'shoulder_rotation': shoulder_rotation,
            'hip_rotation': hip_rotation,
            'posture_change': posture_change,
            'issues': issues
        }
    
    def _analyze_retraction(self, frames: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Analyze the retraction phase of the jab
        
        Args:
            frames: Frames of the jab
            
        Returns:
            Analysis of retraction
        """
        if not frames or len(frames) < 5:
            return {'quality': 0, 'issues': ['Not enough frames for retraction analysis']}
        
        # Get the extension and retraction phases
        mid_point = len(frames) // 2
        extension_frames = frames[:mid_point+1]
        retraction_frames = frames[mid_point:]
        
        # Check if we have all required landmarks
        if not all('right_wrist' in frame for frame in frames):
            return {'quality': 0, 'issues': ['Missing wrist positions for retraction analysis']}
        
        # Track wrist positions
        extension_positions = []
        for frame in extension_frames:
            extension_positions.append(np.array([
                frame['right_wrist']['x'],
                frame['right_wrist']['y'],
                frame['right_wrist']['z']
            ]))
        
        retraction_positions = []
        for frame in retraction_frames:
            retraction_positions.append(np.array([
                frame['right_wrist']['x'],
                frame['right_wrist']['y'],
                frame['right_wrist']['z']
            ]))
        
        # Reverse retraction positions to compare paths
        retraction_positions.reverse()
        
        # Calculate path similarity between extension and retraction
        # (should follow same path in reverse)
        min_length = min(len(extension_positions), len(retraction_positions))
        path_diffs = []
        
        for i in range(min_length):
            diff = np.linalg.norm(extension_positions[i] - retraction_positions[i])
            path_diffs.append(diff)
        
        avg_path_diff = np.mean(path_diffs) if path_diffs else float('inf')
        
        # Calculate retraction speed (should be fast)
        retraction_time = len(retraction_frames)
        extension_time = len(extension_frames)
        
        # Speed ratio (retraction should be similar or faster than extension)
        speed_ratio = extension_time / retraction_time if retraction_time > 0 else 0
        
        # Check final position (should return to guard)
        if 'nose' in frames[0] and 'right_wrist' in frames[0] and 'right_wrist' in frames[-1]:
            start_nose_to_wrist = np.linalg.norm(
                np.array([frames[0]['nose']['x'], frames[0]['nose']['y']]) - 
                np.array([frames[0]['right_wrist']['x'], frames[0]['right_wrist']['y']])
            )
            
            end_nose_to_wrist = np.linalg.norm(
                np.array([frames[0]['nose']['x'], frames[0]['nose']['y']]) - 
                np.array([frames[-1]['right_wrist']['x'], frames[-1]['right_wrist']['y']])
            )
            
            # Return to guard score (1.0 = perfect return)
            return_score = max(0, 1.0 - abs(end_nose_to_wrist - start_nose_to_wrist) / start_nose_to_wrist)
        else:
            return_score = 0.5  # Default if we can't calculate
        
        # Evaluate retraction
        issues = []
        quality = 1.0  # Start with perfect score
        
        # Check path similarity
        path_score = max(0, 1.0 - avg_path_diff)
        if avg_path_diff > 0.1:
            issues.append('Retraction path differs from extension path')
            quality -= 0.3
        
        # Check retraction speed
        speed_score = min(1.0, speed_ratio)
        if speed_ratio < 0.8:
            issues.append('Slow retraction (should be as fast or faster than extension)')
            quality -= 0.3
        
        # Check return to guard
        if return_score < 0.7:
            issues.append('Poor return to guard position')
            quality -= 0.3
        
        return {
            'quality': max(0, quality),
            'path_score': path_score,
            'speed_score': speed_score,
            'return_score': return_score,
            'path_difference': avg_path_diff,
            'speed_ratio': speed_ratio,
            'issues': issues
        }
    
    def _detect_common_errors(self, frames: List[Dict[str, Dict[str, float]]], 
                              extension_trajectory: Dict[str, Any], 
                              retraction: Dict[str, Any]) -> Dict[str, bool]:
        """
        Detect common jab errors
        
        Args:
            frames: Frames of the jab
            extension_trajectory: Extension trajectory analysis
            retraction: Retraction analysis
            
        Returns:
            Dictionary of detected errors
        """
        if not frames or len(frames) < 5:
            return {}
        
        errors = {
            'telegraphing': False,
            'looping': False,
            'pawing': False,
            'overextending': False,
            'dropping_guard': False,
            'poor_retraction': False
        }
        
        # Check for telegraphing (dropping hand before extending)
        if len(frames) >= 3 and 'right_wrist' in frames[0] and 'right_shoulder' in frames[0]:
            start_wrist = np.array([frames[0]['right_wrist']['y']])
            pre_extend_wrist = np.array([frames[2]['right_wrist']['y']])
            
            # If wrist drops before extension
            if pre_extend_wrist > start_wrist + 0.05:  # Y increases as you go down
                errors['telegraphing'] = True
        
        # Check for looping (non-straight trajectory)
        if 'straightness_ratio' in extension_trajectory and extension_trajectory['straightness_ratio'] > 1.2:
            errors['looping'] = True
        
        # Also check for looping in the issues
        if 'issues' in extension_trajectory:
            for issue in extension_trajectory['issues']:
                if 'trajectory' in issue.lower() and ('not straight' in issue.lower() or 'looping' in issue.lower()):
                    errors['looping'] = True
        
        # Check for pawing (incomplete extension)
        if 'extension_angle' in extension_trajectory and extension_trajectory['extension_angle'] < 150:
            errors['pawing'] = True
        
        # Also check for pawing in the issues
        if 'issues' in extension_trajectory:
            for issue in extension_trajectory['issues']:
                if 'incomplete' in issue.lower() or 'not fully committing' in issue.lower():
                    errors['pawing'] = True
        
        # Check for overextending
        if 'extension_angle' in extension_trajectory and extension_trajectory['extension_angle'] > 190:
            errors['overextending'] = True
        
        # Also check for overextending in the issues
        if 'issues' in extension_trajectory:
            for issue in extension_trajectory['issues']:
                if 'overextend' in issue.lower() or 'hyperextend' in issue.lower():
                    errors['overextending'] = True
        
        # Check for dropping guard (opposite hand)
        # This requires tracking the non-jabbing hand
        mid_point = len(frames) // 2
        if 'left_wrist' in frames[0] and 'left_shoulder' in frames[0] and 'left_wrist' in frames[mid_point]:
            start_guard_height = frames[0]['left_wrist']['y'] - frames[0]['left_shoulder']['y']
            mid_guard_height = frames[mid_point]['left_wrist']['y'] - frames[mid_point]['left_shoulder']['y']
            
            # If guard hand drops during jab
            if mid_guard_height > start_guard_height + 0.1:  # Y increases as you go down
                errors['dropping_guard'] = True
        
        # Check for poor retraction
        if 'quality' in retraction and retraction['quality'] < 0.6:
            errors['poor_retraction'] = True
        
        # Also check for poor retraction in the issues
        if 'issues' in retraction:
            for issue in retraction['issues']:
                if 'slow' in issue.lower() or 'path' in issue.lower() or 'guard' in issue.lower():
                    errors['poor_retraction'] = True
        
        return errors
    
    def _evaluate_technical_elements(self, enhanced_features: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate technical elements across all jabs
        
        Args:
            enhanced_features: Enhanced features for each jab
            
        Returns:
            Technical scores
        """
        if not enhanced_features:
            return {}
        
        # Initialize scores
        scores = {
            'starting_position': 0,
            'extension_trajectory': 0,
            'hand_position': 0,
            'body_mechanics': 0,
            'retraction': 0,
            'overall_technique': 0
        }
        
        # Calculate average scores across all jabs
        for feature in enhanced_features:
            scores['starting_position'] += feature['starting_position']['quality']
            scores['extension_trajectory'] += feature['extension_trajectory']['quality']
            scores['hand_position'] += feature['hand_position']['quality']
            scores['body_mechanics'] += feature['body_mechanics']['quality']
            scores['retraction'] += feature['retraction']['quality']
        
        # Average the scores
        num_jabs = len(enhanced_features)
        for key in scores:
            if key != 'overall_technique':
                scores[key] /= num_jabs
        
        # Calculate overall technique score (weighted average)
        weights = {
            'starting_position': 0.15,
            'extension_trajectory': 0.3,
            'hand_position': 0.15,
            'body_mechanics': 0.2,
            'retraction': 0.2
        }
        
        scores['overall_technique'] = sum(scores[key] * weights[key] for key in weights)
        
        # Convert to percentages
        for key in scores:
            scores[key] = round(scores[key] * 100, 1)
        
        return scores
