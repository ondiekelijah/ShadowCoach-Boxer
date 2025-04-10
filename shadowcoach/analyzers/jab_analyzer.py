"""
Jab technique analyzer for the ShadowCoach system.

This module provides detailed analysis of boxing jab techniques using pose data.
It implements advanced algorithms for technique detection, comparison, and scoring.

Key Features:
    - Jab detection using pose data
    - Orientation-invariant feature extraction
    - Dynamic time warping for sequence comparison
    - Detailed metrics calculation
    - Reference technique comparison

Technical Details:
    - Uses normalized features to handle different body sizes
    - Implements DTW for temporal alignment of sequences
    - Provides configurable thresholds for detection
    - Handles both right-handed and left-handed techniques

Algorithm Overview:
    1. Extract normalized features from pose data
    2. Detect potential jabs using speed and extension patterns
    3. Analyze each jab for metrics (speed, extension, duration)
    4. Compare to reference using DTW if available
    5. Generate detailed analysis results
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.signal import savgol_filter

from ..utils.logging_utils import logger, timed
from ..core.reference_manager import ReferenceManager

class JabAnalyzer:
    """
    Analyzer for boxing jab techniques.

    This class provides comprehensive analysis of jab techniques by processing
    pose data and comparing it to reference techniques.

    Key Features:
        - Automated jab detection
        - Feature extraction and normalization
        - Reference comparison
        - Detailed metrics calculation

    Attributes:
        reference_manager: Manager for reference techniques

    Example:
        >>> analyzer = JabAnalyzer(reference_manager)
        >>> analysis = analyzer.analyze_jab(pose_data, "pro_boxer")
        >>> print(f"Detected {len(analysis['jabs'])} jabs")
    """
    def __init__(self, reference_manager: ReferenceManager):
        """
        Initialize the jab analyzer

        Args:
            reference_manager: Reference manager for comparing techniques
        """
        logger.info("Initializing JabAnalyzer...")
        self.reference_manager = reference_manager
        logger.info("JabAnalyzer initialized successfully")

    @timed
    def analyze_jab(self, pose_data: List[Dict[str, Dict[str, float]]],
                   reference_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze jab technique by comparing to reference

        Args:
            pose_data: The pose data to analyze
            reference_name: Optional reference name to compare against

        Returns:
            Analysis metrics
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
        if reference_name:
            reference = self.reference_manager.get_reference(reference_name)
            if reference:
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

    def _detect_jabs(self, pose_data: List[Dict[str, Dict[str, float]]]) -> Optional[Dict[str, Any]]:
        """
        Extract features and detect jabs from pose data

        Args:
            pose_data: The pose data to analyze

        Returns:
            Detected jabs and analysis data
        """
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

    def _calculate_jab_speeds(self, normalized_features: List[Dict[str, float]]) -> List[float]:
        """
        Calculate jab speeds from normalized features

        Args:
            normalized_features: The normalized features

        Returns:
            List of jab speeds
        """
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

    def _identify_potential_jabs(self, normalized_features: List[Dict[str, float]],
                               jab_speeds: List[float],
                               arm_angles: List[float]) -> List[Dict[str, Any]]:
        """
        Identify potential jabs from speed and extension patterns

        Args:
            normalized_features: The normalized features
            jab_speeds: The calculated jab speeds
            arm_angles: The arm angles

        Returns:
            List of detected jabs
        """
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

    def _analyze_single_jab(self, normalized_features: List[Dict[str, float]],
                          jab_speeds: List[float],
                          arm_angles: List[float],
                          start: int,
                          end: int) -> Optional[Dict[str, Any]]:
        """
        Analyze a single potential jab and return its metrics if valid

        Args:
            normalized_features: The normalized features
            jab_speeds: The calculated jab speeds
            arm_angles: The arm angles
            start: Start frame of the jab
            end: End frame of the jab

        Returns:
            Jab metrics if valid, None otherwise
        """
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

    def _calculate_average_metrics(self, jabs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate average metrics across all jabs

        Args:
            jabs: List of detected jabs

        Returns:
            Average metrics
        """
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

    def _compare_to_reference(self, pose_data: List[Dict[str, Dict[str, float]]],
                            user_analysis: Dict[str, Any],
                            reference_name: str) -> Optional[Dict[str, Any]]:
        """
        Compare user jabs to reference model

        Args:
            pose_data: The user's pose data
            user_analysis: The user's jab analysis
            reference_name: The name of the reference to compare against

        Returns:
            Comparison metrics
        """
        logger.info(f"Comparing to reference model: {reference_name}")
        reference = self.reference_manager.get_reference(reference_name)
        if not reference:
            logger.warning(f"Reference '{reference_name}' not found")
            return None

        ref_data = reference['poses']

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

    def _compare_individual_jabs(self, user_data: List[Dict[str, Dict[str, float]]],
                               user_jabs: List[Dict[str, Any]],
                               ref_data: List[Dict[str, Dict[str, float]]],
                               ref_jabs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compare each user jab to reference jabs and find best matches

        Args:
            user_data: The user's pose data
            user_jabs: The user's detected jabs
            ref_data: The reference pose data
            ref_jabs: The reference detected jabs

        Returns:
            List of jab similarities
        """
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

    def _calculate_metrics_comparison(self, user_metrics: Dict[str, Any],
                                    ref_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comparison ratios between user and reference metrics

        Args:
            user_metrics: The user's metrics
            ref_metrics: The reference metrics

        Returns:
            Comparison ratios
        """
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

    def _extract_normalized_features(self, sequence: List[Dict[str, Dict[str, float]]]) -> List[Dict[str, float]]:
        """
        Extract normalized features that are invariant to video orientation

        Args:
            sequence: The pose sequence

        Returns:
            List of normalized features
        """
        features = []

        for frame in sequence:
            # Check if we have all the required landmarks
            required_landmarks = ['left_shoulder', 'right_shoulder', 'left_elbow',
                                'right_elbow', 'left_wrist', 'right_wrist']

            if not all(landmark in frame for landmark in required_landmarks):
                continue

            # Extract landmarks
            l_shoulder = np.array([
                frame['left_shoulder']['x'],
                frame['left_shoulder']['y'],
                frame['left_shoulder']['z']
            ])

            r_shoulder = np.array([
                frame['right_shoulder']['x'],
                frame['right_shoulder']['y'],
                frame['right_shoulder']['z']
            ])

            l_elbow = np.array([
                frame['left_elbow']['x'],
                frame['left_elbow']['y'],
                frame['left_elbow']['z']
            ])

            r_elbow = np.array([
                frame['right_elbow']['x'],
                frame['right_elbow']['y'],
                frame['right_elbow']['z']
            ])

            l_wrist = np.array([
                frame['left_wrist']['x'],
                frame['left_wrist']['y'],
                frame['left_wrist']['z']
            ])

            r_wrist = np.array([
                frame['right_wrist']['x'],
                frame['right_wrist']['y'],
                frame['right_wrist']['z']
            ])

            # Calculate shoulder width (for normalization)
            shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)

            # Calculate arm vectors
            l_upper_arm = l_elbow - l_shoulder
            l_forearm = l_wrist - l_elbow

            r_upper_arm = r_elbow - r_shoulder
            r_forearm = r_wrist - r_elbow

            # Calculate arm angles
            l_cos_angle = np.dot(l_upper_arm, l_forearm) / (
                np.linalg.norm(l_upper_arm) * np.linalg.norm(l_forearm)
            )
            l_angle = np.degrees(np.arccos(np.clip(l_cos_angle, -1.0, 1.0)))

            r_cos_angle = np.dot(r_upper_arm, r_forearm) / (
                np.linalg.norm(r_upper_arm) * np.linalg.norm(r_forearm)
            )
            r_angle = np.degrees(np.arccos(np.clip(r_cos_angle, -1.0, 1.0)))

            # Calculate normalized arm extensions
            l_extension = np.linalg.norm(l_wrist - l_shoulder) / shoulder_width
            r_extension = np.linalg.norm(r_wrist - r_shoulder) / shoulder_width

            # Calculate arm directions (normalized)
            l_direction = (l_wrist - l_shoulder) / shoulder_width
            r_direction = (r_wrist - r_shoulder) / shoulder_width

            # Store features
            feature = {
                'l_angle': l_angle,
                'r_angle': r_angle,
                'l_extension': l_extension,
                'r_extension': r_extension,
                'l_direction_x': l_direction[0],
                'l_direction_y': l_direction[1],
                'r_direction_x': r_direction[0],
                'r_direction_y': r_direction[1],
                'shoulder_width': shoulder_width
            }

            features.append(feature)

        return features

    def _compare_sequences(self, seq1: List[Dict[str, Dict[str, float]]],
                         seq2: List[Dict[str, Dict[str, float]]]) -> float:
        """
        Compare two pose sequences and return similarity score (0-1)

        Args:
            seq1: First pose sequence
            seq2: Second pose sequence

        Returns:
            Similarity score between 0 and 1
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

    def _dtw_similarity(self, seq1: List[Dict[str, float]],
                      seq2: List[Dict[str, float]],
                      max_warping: float = 0.5) -> float:
        """
        Calculate similarity between sequences using Dynamic Time Warping

        Args:
            seq1: First feature sequence
            seq2: Second feature sequence
            max_warping: Maximum warping allowed (as a fraction of sequence length)

        Returns:
            Similarity score between 0 and 1
        """
        n, m = len(seq1), len(seq2)

        # Calculate maximum warping distance
        w = int(max(n, m) * max_warping)

        # Initialize cost matrix
        dtw = np.zeros((n+1, m+1))
        dtw[0, 1:] = np.inf
        dtw[1:, 0] = np.inf

        # Fill the cost matrix
        for i in range(1, n+1):
            # Limit the warping window
            for j in range(max(1, i-w), min(m+1, i+w+1)):
                cost = self._feature_distance(seq1[i-1], seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

        # Calculate similarity score (inverse of normalized distance)
        # The smaller the DTW distance, the more similar the sequences
        distance = dtw[n, m]

        # Normalize by sequence length
        normalized_distance = distance / (n + m)

        # Convert to similarity score (0-1)
        # Using exponential decay: similarity = e^(-distance)
        similarity = np.exp(-normalized_distance)

        return similarity

    def _feature_distance(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        Calculate distance between two feature sets

        Args:
            features1: First feature set
            features2: Second feature set

        Returns:
            Distance between the features
        """
        # Define weights for different features
        weights = {
            'angle': 0.3,        # Arm angle is important but can vary
            'extension': 0.4,    # Extension is very important for jabs
            'direction': 0.3     # Direction helps with orientation invariance
        }

        # Calculate weighted distance for each feature type
        angle_dist = (
            abs(features1['r_angle'] - features2['r_angle']) / 180.0 +
            abs(features1['l_angle'] - features2['l_angle']) / 180.0
        ) / 2.0

        extension_dist = (
            abs(features1['r_extension'] - features2['r_extension']) +
            abs(features1['l_extension'] - features2['l_extension'])
        ) / 2.0

        direction_dist = (
            np.sqrt(
                (features1['r_direction_x'] - features2['r_direction_x'])**2 +
                (features1['r_direction_y'] - features2['r_direction_y'])**2
            ) +
            np.sqrt(
                (features1['l_direction_x'] - features2['l_direction_x'])**2 +
                (features1['l_direction_y'] - features2['l_direction_y'])**2
            )
        ) / 2.0

        # Combine distances with weights
        weighted_distance = (
            weights['angle'] * angle_dist +
            weights['extension'] * extension_dist +
            weights['direction'] * direction_dist
        )

        return weighted_distance
