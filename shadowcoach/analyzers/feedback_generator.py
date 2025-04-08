"""
Feedback generation for boxing technique analysis
"""
from typing import Dict, List, Any, Optional

from ..utils.logging_utils import logger, timed

class FeedbackGenerator:
    """
    Class for generating feedback based on boxing technique analysis
    """
    def __init__(self):
        """Initialize the feedback generator"""
        logger.info("Initializing FeedbackGenerator...")
        logger.info("FeedbackGenerator initialized successfully")
    
    @timed
    def generate_feedback(self, user_analysis: Dict[str, Any], 
                         ref_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate detailed feedback based on comparison to reference
        
        Args:
            user_analysis: Analysis of the user's technique
            ref_analysis: Analysis of the reference technique
            
        Returns:
            List of feedback points
        """
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
    
    def _has_valid_jab_data(self, user_analysis: Dict[str, Any], 
                           ref_analysis: Dict[str, Any], 
                           feedback: List[str]) -> bool:
        """
        Check if we have valid jab data to generate feedback
        
        Args:
            user_analysis: Analysis of the user's technique
            ref_analysis: Analysis of the reference technique
            feedback: List to add feedback to
            
        Returns:
            True if valid data is available, False otherwise
        """
        if not user_analysis['jabs']:
            feedback.append("No jabs detected in your video. Try performing clearer jab movements.")
            return False
            
        if not ref_analysis['jabs']:
            feedback.append("No jabs detected in reference video. Please select a different reference.")
            return False
            
        return True
    
    def _get_metrics_comparison(self, comparison: Optional[Dict[str, Any]], 
                              feedback: List[str]) -> Optional[Dict[str, float]]:
        """
        Get metrics comparison data if available
        
        Args:
            comparison: Comparison data
            feedback: List to add feedback to
            
        Returns:
            Metrics comparison data or None if not available
        """
        if not comparison:
            feedback.append("Could not compare to reference. Try recording with better lighting and camera angle.")
            return None
            
        metrics_comp = comparison.get('metrics_comparison', None)
        if not metrics_comp:
            feedback.append("Could not calculate comparison metrics. Try recording with better lighting and camera angle.")
            return None
            
        return metrics_comp
    
    def _add_jab_count_feedback(self, user_analysis: Dict[str, Any], 
                              ref_analysis: Dict[str, Any], 
                              feedback: List[str]) -> None:
        """
        Add feedback about jab count
        
        Args:
            user_analysis: Analysis of the user's technique
            ref_analysis: Analysis of the reference technique
            feedback: List to add feedback to
        """
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
    
    def _add_duration_feedback(self, metrics_comp: Dict[str, float], 
                             feedback: List[str]) -> None:
        """
        Add feedback about jab duration
        
        Args:
            metrics_comp: Metrics comparison data
            feedback: List to add feedback to
        """
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
    
    def _add_speed_feedback(self, metrics_comp: Dict[str, float], 
                          feedback: List[str]) -> None:
        """
        Add feedback about jab speed
        
        Args:
            metrics_comp: Metrics comparison data
            feedback: List to add feedback to
        """
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
    
    def _add_extension_feedback(self, metrics_comp: Dict[str, float], 
                              feedback: List[str]) -> None:
        """
        Add feedback about arm extension
        
        Args:
            metrics_comp: Metrics comparison data
            feedback: List to add feedback to
        """
        extension_ratio = metrics_comp.get('extension_ratio', 0)
        norm_extension_ratio = metrics_comp.get('norm_extension_ratio', 0)
        
        # Use normalized extension if available, otherwise use raw extension
        if norm_extension_ratio > 0:
            self._add_extension_ratio_feedback(norm_extension_ratio, feedback)
        elif extension_ratio > 0:
            self._add_extension_ratio_feedback(extension_ratio, feedback)
    
    def _add_extension_ratio_feedback(self, ratio: float, 
                                    feedback: List[str]) -> None:
        """
        Add feedback based on extension ratio
        
        Args:
            ratio: Extension ratio
            feedback: List to add feedback to
        """
        if ratio < 0.8:
            feedback.append("You're not fully extending your arm during jabs. Try to extend more "
                           "while maintaining proper form and guard position.")
        elif ratio > 1.2:
            feedback.append("You may be over-extending your jabs. Be careful not to hyperextend "
                           "your elbow or compromise your guard.")
        else:
            feedback.append("Good job with your jab extension. You're reaching the optimal distance.")
    
    def _add_similarity_feedback(self, comparison: Dict[str, Any], 
                               feedback: List[str]) -> None:
        """
        Add feedback about overall similarity
        
        Args:
            comparison: Comparison data
            feedback: List to add feedback to
        """
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
    
    def _add_specific_jab_feedback(self, comparison: Dict[str, Any], 
                                 user_analysis: Dict[str, Any], 
                                 feedback: List[str]) -> None:
        """
        Add feedback about specific jabs
        
        Args:
            comparison: Comparison data
            user_analysis: Analysis of the user's technique
            feedback: List to add feedback to
        """
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
    
    def _add_summary_feedback(self, metrics_comp: Dict[str, float], 
                            feedback: List[str]) -> None:
        """
        Add summary feedback with overall score
        
        Args:
            metrics_comp: Metrics comparison data
            feedback: List to add feedback to
        """
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
