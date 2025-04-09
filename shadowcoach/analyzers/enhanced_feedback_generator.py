"""
Enhanced feedback generation for boxing technique analysis
"""
from typing import Dict, List, Any, Optional

from ..utils.logging_utils import logger, timed
from .feedback_generator import FeedbackGenerator

class EnhancedFeedbackGenerator(FeedbackGenerator):
    """
    Enhanced class for generating detailed feedback based on boxing fundamentals
    """
    def __init__(self):
        """Initialize the enhanced feedback generator"""
        super().__init__()
        logger.info("Initializing EnhancedFeedbackGenerator...")
        logger.info("EnhancedFeedbackGenerator initialized successfully")
    
    @timed
    def generate_feedback(self, user_analysis: Dict[str, Any], 
                         ref_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate detailed feedback based on enhanced analysis
        
        Args:
            user_analysis: Analysis of the user's technique
            ref_analysis: Analysis of the reference technique
            
        Returns:
            List of feedback points
        """
        # Get basic feedback from parent class
        basic_feedback = super().generate_feedback(user_analysis, ref_analysis)
        
        # If no enhanced features are available, return basic feedback
        if 'enhanced_features' not in user_analysis or not user_analysis['enhanced_features']:
            return basic_feedback
        
        # Generate enhanced feedback
        enhanced_feedback = []
        
        # Add technical scores feedback
        if 'technical_scores' in user_analysis:
            enhanced_feedback.extend(self._generate_technical_scores_feedback(user_analysis['technical_scores']))
        
        # Add detailed feedback for each technical element
        enhanced_feedback.extend(self._generate_starting_position_feedback(user_analysis['enhanced_features']))
        enhanced_feedback.extend(self._generate_extension_trajectory_feedback(user_analysis['enhanced_features']))
        enhanced_feedback.extend(self._generate_body_mechanics_feedback(user_analysis['enhanced_features']))
        enhanced_feedback.extend(self._generate_retraction_feedback(user_analysis['enhanced_features']))
        
        # Add common errors feedback
        enhanced_feedback.extend(self._generate_common_errors_feedback(user_analysis['enhanced_features']))
        
        # Combine basic and enhanced feedback
        # Remove any duplicate or redundant feedback
        combined_feedback = basic_feedback.copy()
        for feedback in enhanced_feedback:
            if not any(self._is_similar_feedback(feedback, existing) for existing in combined_feedback):
                combined_feedback.append(feedback)
        
        return combined_feedback
    
    def _is_similar_feedback(self, feedback1: str, feedback2: str) -> bool:
        """
        Check if two feedback points are similar
        
        Args:
            feedback1: First feedback point
            feedback2: Second feedback point
            
        Returns:
            True if the feedback points are similar
        """
        # Simple similarity check based on common words
        words1 = set(feedback1.lower().split())
        words2 = set(feedback2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        
        return similarity > 0.5  # Threshold for similarity
    
    def _generate_technical_scores_feedback(self, scores: Dict[str, float]) -> List[str]:
        """
        Generate feedback based on technical scores
        
        Args:
            scores: Technical scores
            
        Returns:
            List of feedback points
        """
        feedback = []
        
        # Add overall technique score
        overall_score = scores.get('overall_technique', 0)
        if overall_score >= 90:
            feedback.append(f"Excellent technique! Your overall technical score is {overall_score}%. "
                           f"You're demonstrating professional-level jab mechanics.")
        elif overall_score >= 80:
            feedback.append(f"Very good technique. Your overall technical score is {overall_score}%. "
                           f"You're showing solid fundamentals with a few areas for improvement.")
        elif overall_score >= 70:
            feedback.append(f"Good technique. Your overall technical score is {overall_score}%. "
                           f"You have a solid foundation but several areas need refinement.")
        elif overall_score >= 60:
            feedback.append(f"Fair technique. Your overall technical score is {overall_score}%. "
                           f"You're showing basic understanding but need significant improvement in multiple areas.")
        else:
            feedback.append(f"Your overall technical score is {overall_score}%. "
                           f"Focus on mastering the fundamental mechanics of the jab.")
        
        # Identify strongest and weakest areas
        score_items = [(key, value) for key, value in scores.items() if key != 'overall_technique']
        if score_items:
            score_items.sort(key=lambda x: x[1], reverse=True)
            
            # Format the technical element names for readability
            element_names = {
                'starting_position': 'starting position',
                'extension_trajectory': 'extension and trajectory',
                'hand_position': 'hand position',
                'body_mechanics': 'body mechanics',
                'retraction': 'retraction'
            }
            
            # Strongest area
            strongest = score_items[0]
            if strongest[1] >= 70:
                feedback.append(f"Your strongest aspect is {element_names.get(strongest[0], strongest[0])} "
                               f"with a score of {strongest[1]}%.")
            
            # Weakest area
            weakest = score_items[-1]
            if weakest[1] < 70:
                feedback.append(f"Your main area for improvement is {element_names.get(weakest[0], weakest[0])} "
                               f"with a score of {weakest[1]}%. Focus on this aspect in your training.")
        
        return feedback
    
    def _generate_starting_position_feedback(self, enhanced_features: List[Dict[str, Any]]) -> List[str]:
        """
        Generate feedback on starting position
        
        Args:
            enhanced_features: Enhanced features for each jab
            
        Returns:
            List of feedback points
        """
        feedback = []
        
        # Collect all issues across jabs
        all_issues = []
        for feature in enhanced_features:
            starting_position = feature.get('starting_position', {})
            issues = starting_position.get('issues', [])
            all_issues.extend(issues)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Generate feedback for common issues
        for issue, count in issue_counts.items():
            if count >= len(enhanced_features) / 2:  # Issue occurs in at least half of jabs
                if "guard position" in issue.lower():
                    feedback.append(f"Starting Position: {issue}. Keep your hand close to your chin in the guard position.")
                elif "elbow" in issue.lower():
                    feedback.append(f"Starting Position: {issue}. Keep your elbow tucked to protect your ribs.")
                else:
                    feedback.append(f"Starting Position: {issue}")
        
        return feedback
    
    def _generate_extension_trajectory_feedback(self, enhanced_features: List[Dict[str, Any]]) -> List[str]:
        """
        Generate feedback on extension and trajectory
        
        Args:
            enhanced_features: Enhanced features for each jab
            
        Returns:
            List of feedback points
        """
        feedback = []
        
        # Collect all issues across jabs
        all_issues = []
        for feature in enhanced_features:
            extension = feature.get('extension_trajectory', {})
            issues = extension.get('issues', [])
            all_issues.extend(issues)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Generate feedback for common issues
        for issue, count in issue_counts.items():
            if count >= len(enhanced_features) / 2:  # Issue occurs in at least half of jabs
                if "trajectory" in issue.lower() or "looping" in issue.lower():
                    feedback.append(f"Extension: {issue}. Focus on extending your jab in a straight line directly to the target.")
                elif "extension" in issue.lower() and "incomplete" in issue.lower():
                    feedback.append(f"Extension: {issue}. Fully commit to your jab by extending your arm completely.")
                elif "hyperextending" in issue.lower() or "overextending" in issue.lower():
                    feedback.append(f"Extension: {issue}. Be careful not to lock your elbow at full extension.")
                else:
                    feedback.append(f"Extension: {issue}")
        
        return feedback
    
    def _generate_body_mechanics_feedback(self, enhanced_features: List[Dict[str, Any]]) -> List[str]:
        """
        Generate feedback on body mechanics
        
        Args:
            enhanced_features: Enhanced features for each jab
            
        Returns:
            List of feedback points
        """
        feedback = []
        
        # Collect all issues across jabs
        all_issues = []
        for feature in enhanced_features:
            mechanics = feature.get('body_mechanics', {})
            issues = mechanics.get('issues', [])
            all_issues.extend(issues)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Generate feedback for common issues
        for issue, count in issue_counts.items():
            if count >= len(enhanced_features) / 2:  # Issue occurs in at least half of jabs
                if "shoulder rotation" in issue.lower() and "insufficient" in issue.lower():
                    feedback.append(f"Body Mechanics: {issue}. Rotate your shoulder slightly forward to protect your chin.")
                elif "shoulder rotation" in issue.lower() and "excessive" in issue.lower():
                    feedback.append(f"Body Mechanics: {issue}. Don't overcommit with your shoulder rotation.")
                elif "hip rotation" in issue.lower() and "insufficient" in issue.lower():
                    feedback.append(f"Body Mechanics: {issue}. Engage your hips slightly to add power to your jab.")
                elif "hip rotation" in issue.lower() and "excessive" in issue.lower():
                    feedback.append(f"Body Mechanics: {issue}. Limit your hip rotation to maintain balance.")
                elif "posture" in issue.lower():
                    feedback.append(f"Body Mechanics: {issue}. Maintain your posture throughout the jab motion.")
                else:
                    feedback.append(f"Body Mechanics: {issue}")
        
        return feedback
    
    def _generate_retraction_feedback(self, enhanced_features: List[Dict[str, Any]]) -> List[str]:
        """
        Generate feedback on retraction
        
        Args:
            enhanced_features: Enhanced features for each jab
            
        Returns:
            List of feedback points
        """
        feedback = []
        
        # Collect all issues across jabs
        all_issues = []
        for feature in enhanced_features:
            retraction = feature.get('retraction', {})
            issues = retraction.get('issues', [])
            all_issues.extend(issues)
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Generate feedback for common issues
        for issue, count in issue_counts.items():
            if count >= len(enhanced_features) / 2:  # Issue occurs in at least half of jabs
                if "path" in issue.lower():
                    feedback.append(f"Retraction: {issue}. Return your hand along the same path it traveled during extension.")
                elif "speed" in issue.lower() or "slow" in issue.lower():
                    feedback.append(f"Retraction: {issue}. Retract your jab as quickly as you extend it.")
                elif "guard" in issue.lower():
                    feedback.append(f"Retraction: {issue}. Always return your hand to the guard position to protect yourself.")
                else:
                    feedback.append(f"Retraction: {issue}")
        
        return feedback
    
    def _generate_common_errors_feedback(self, enhanced_features: List[Dict[str, Any]]) -> List[str]:
        """
        Generate feedback on common errors
        
        Args:
            enhanced_features: Enhanced features for each jab
            
        Returns:
            List of feedback points
        """
        feedback = []
        
        # Collect all errors across jabs
        error_counts = {
            'telegraphing': 0,
            'looping': 0,
            'pawing': 0,
            'overextending': 0,
            'dropping_guard': 0,
            'poor_retraction': 0
        }
        
        for feature in enhanced_features:
            errors = feature.get('common_errors', {})
            for error, detected in errors.items():
                if detected:
                    error_counts[error] = error_counts.get(error, 0) + 1
        
        # Generate feedback for common errors
        num_jabs = len(enhanced_features)
        error_threshold = num_jabs / 3  # Error occurs in at least a third of jabs
        
        error_messages = {
            'telegraphing': "You're telegraphing your jab by dropping your hand before extending. "
                           "This gives your opponent time to react. Keep your hand in guard until you extend.",
            
            'looping': "Your jab is taking a curved path rather than traveling in a straight line. "
                      "This makes it slower and easier to see coming. Practice extending directly from your guard.",
            
            'pawing': "You're 'pawing' with your jab rather than fully committing to it. "
                     "This reduces its effectiveness. Fully extend your arm and commit to the punch.",
            
            'overextending': "You're overextending on your jab, which can compromise your balance and leave you vulnerable. "
                            "Extend to your target without leaning or reaching too far.",
            
            'dropping_guard': "You're dropping your rear hand when jabbing, leaving your chin exposed. "
                             "Keep your rear hand up to protect yourself at all times.",
            
            'poor_retraction': "Your jab retraction needs improvement. Return your hand quickly along the same path "
                              "to your guard position after extending."
        }
        
        for error, count in error_counts.items():
            if count >= error_threshold and error in error_messages:
                feedback.append(error_messages[error])
        
        return feedback

    @timed
    def generate_comparative_feedback(self, user_analysis: Dict[str, Any], 
                                     reference_analyses: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate comparative feedback across multiple references
        
        Args:
            user_analysis: Analysis of the user's technique
            reference_analyses: Dictionary of reference analyses keyed by reference name
            
        Returns:
            List of feedback points
        """
        if not reference_analyses:
            return ["No reference analyses provided for comparison."]
        
        feedback = []
        
        # Get primary reference feedback
        primary_ref = list(reference_analyses.keys())[0]
        primary_feedback = self.generate_feedback(user_analysis, reference_analyses[primary_ref])
        feedback.extend(primary_feedback)
        
        # If only one reference, return primary feedback
        if len(reference_analyses) == 1:
            return feedback
        
        # Add comparative feedback header
        feedback.append("\nComparative Analysis:")
        
        # Compare similarity scores across references
        similarity_scores = {}
        for ref_name, ref_analysis in reference_analyses.items():
            if user_analysis['reference_comparison'] and user_analysis['reference_comparison']['average_similarity'] > 0:
                similarity_scores[ref_name] = user_analysis['reference_comparison']['average_similarity'] * 100
        
        if similarity_scores:
            # Find best and worst matches
            best_match = max(similarity_scores.items(), key=lambda x: x[1])
            worst_match = min(similarity_scores.items(), key=lambda x: x[1])
            
            # Add feedback about best match
            feedback.append(f"Your technique most closely matches the '{best_match[0]}' reference " +
                           f"with {best_match[1]:.1f}% similarity.")
            
            # Add feedback about style differences if there's a significant gap
            if best_match[1] - worst_match[1] > 15:  # More than 15% difference
                feedback.append(f"Your technique differs significantly from the '{worst_match[0]}' reference " +
                               f"({worst_match[1]:.1f}% similarity). This suggests your natural style may be " +
                               f"more aligned with the '{best_match[0]}' approach.")
        
        # Compare jab counts
        jab_counts = {ref_name: len(ref_analysis['jabs']) for ref_name, ref_analysis in reference_analyses.items()}
        user_jab_count = len(user_analysis['jabs'])
        
        # Find references with similar jab counts
        similar_count_refs = [ref_name for ref_name, count in jab_counts.items() 
                             if abs(count - user_jab_count) <= 1]
        
        if similar_count_refs:
            if len(similar_count_refs) == 1:
                feedback.append(f"Your jab frequency is similar to the '{similar_count_refs[0]}' reference.")
            else:
                feedback.append(f"Your jab frequency is similar to the following references: " +
                               f"{', '.join([f'{r}' for r in similar_count_refs])}.")
        
        # Compare jab speeds if available
        if 'avg_metrics' in user_analysis and all('avg_metrics' in ref for ref in reference_analyses.values()):
            user_speed = user_analysis['avg_metrics'].get('avg_max_speed', 0)
            ref_speeds = {ref_name: ref_analysis['avg_metrics'].get('avg_max_speed', 0) 
                         for ref_name, ref_analysis in reference_analyses.items()}
            
            if user_speed > 0 and all(speed > 0 for speed in ref_speeds.values()):
                # Find fastest reference
                fastest_ref = max(ref_speeds.items(), key=lambda x: x[1])
                
                # Compare user speed to fastest reference
                speed_ratio = user_speed / fastest_ref[1]
                if speed_ratio >= 0.9:  # Within 10% of fastest
                    feedback.append(f"Your jab speed is comparable to the fast '{fastest_ref[0]}' reference.")
                elif speed_ratio >= 0.7:  # Within 30% of fastest
                    feedback.append(f"Your jab speed is moderately slower than the '{fastest_ref[0]}' reference. " +
                                   f"Focus on explosive power to increase speed.")
                else:
                    feedback.append(f"Your jab speed is significantly slower than the '{fastest_ref[0]}' reference. " +
                                   f"Practice speed drills to improve.")
        
        # Add style-specific recommendations
        feedback.append("\nStyle-Specific Recommendations:")
        
        # Find the reference with highest similarity
        if similarity_scores:
            best_match_ref = max(similarity_scores.items(), key=lambda x: x[1])[0]
            feedback.append(f"Since your technique most closely matches the '{best_match_ref}' style, " +
                           f"focus on perfecting the technical elements emphasized in this reference.")
        
        return feedback
