"""
Visualization tools for comparing boxing techniques
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.logging_utils import logger, timed

class ComparisonVisualizer:
    """
    Class for creating visualizations comparing boxing techniques
    """
    def __init__(self):
        """Initialize the comparison visualizer"""
        logger.info("Initializing ComparisonVisualizer...")
        logger.info("ComparisonVisualizer initialized successfully")
    
    @timed
    def visualize_comparison(self, user_analysis: Dict[str, Any], 
                           ref_analysis: Dict[str, Any],
                           user_data: List[Dict[str, Dict[str, float]]],
                           reference_data: List[Dict[str, Dict[str, float]]],
                           output_path: str) -> None:
        """
        Create a visualization comparing user technique to reference
        
        Args:
            user_analysis: Analysis of the user's technique
            ref_analysis: Analysis of the reference technique
            user_data: The user's pose data
            reference_data: The reference pose data
            output_path: Path to save the visualization
        """
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
    
    def _plot_arm_angle_comparison(self, user_analysis: Dict[str, Any], 
                                 ref_analysis: Dict[str, Any],
                                 reference_data: List[Dict[str, Dict[str, float]]],
                                 user_data: List[Dict[str, Dict[str, float]]]) -> None:
        """
        Plot arm angle comparison between user and reference
        
        Args:
            user_analysis: Analysis of the user's technique
            ref_analysis: Analysis of the reference technique
            reference_data: The reference pose data
            user_data: The user's pose data
        """
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
    
    def _plot_jab_speed_comparison(self, user_analysis: Dict[str, Any]) -> None:
        """
        Plot jab speed comparison
        
        Args:
            user_analysis: Analysis of the user's technique
        """
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
    
    def _plot_metrics_comparison(self, user_analysis: Dict[str, Any], 
                               ref_analysis: Dict[str, Any]) -> None:
        """
        Plot metrics comparison between user and reference
        
        Args:
            user_analysis: Analysis of the user's technique
            ref_analysis: Analysis of the reference technique
        """
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
