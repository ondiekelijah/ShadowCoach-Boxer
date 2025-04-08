#!/usr/bin/env python3
"""
ShadowCoach Boxing Analysis System - Main Entry Point
A modular system for analyzing boxing techniques using computer vision
"""
import time
from pathlib import Path

# Import core components
from shadowcoach.core.pose_processor import PoseProcessor
from shadowcoach.core.reference_manager import ReferenceManager

# Import analyzers
from shadowcoach.analyzers.jab_analyzer import JabAnalyzer
from shadowcoach.analyzers.feedback_generator import FeedbackGenerator

# Import visualization
from shadowcoach.visualization.comparison_visualizer import ComparisonVisualizer

# Import utilities
from shadowcoach.utils.logging_utils import logger, print_timing_summary
from shadowcoach.utils.file_utils import generate_unique_filename, ensure_directory_exists

def main():
    """Main entry point for the ShadowCoach Boxing Analysis System"""
    logger.info("=== ShadowCoach Boxing Analysis System ===")
    logger.info("Starting analysis process...")
    
    start_time = time.time()
    
    # Initialize components
    pose_processor = PoseProcessor()
    reference_manager = ReferenceManager()
    jab_analyzer = JabAnalyzer(reference_manager)
    feedback_generator = FeedbackGenerator()
    visualizer = ComparisonVisualizer()
    
    # Define video paths
    reference_video = "assets/How to Throw a Jab.mp4"
    user_video_path = "assets/The Wrong Jab.mp4"
    
    # Create output paths
    reference_output = generate_unique_filename("output", "reference_analyzed", ".mp4")
    user_output = generate_unique_filename("output", "user_analyzed", ".mp4")
    comparison_output = generate_unique_filename("output", "jab_comparison", ".png")
    
    # Create output directories
    ensure_directory_exists("output")
    ensure_directory_exists("models")
    
    # Process reference video
    logger.info(f"Processing reference video: {reference_video}")
    ref_poses = pose_processor.process_video(
        reference_video, 
        output_path=reference_output
    )
    
    # Save reference
    reference_manager.save_reference("pro_boxer", ref_poses, {
        "boxer_name": "Professional Example",
        "style": "Orthodox",
        "technique": "Jab practice"
    })
    
    # Check if user video exists
    user_video_exists = Path(user_video_path).exists()
    
    if user_video_exists:
        logger.info(f"Processing user video: {user_video_path}")
        # Process user video
        user_poses = pose_processor.process_video(
            user_video_path, 
            output_path=user_output
        )
        
        # Analyze user jabs
        logger.info("Analyzing user jabs...")
        user_analysis = jab_analyzer.analyze_jab(user_poses, "pro_boxer")
        
        # Analyze reference jabs
        logger.info("Analyzing reference jabs...")
        ref_analysis = jab_analyzer.analyze_jab(ref_poses)
        
        # Generate feedback
        logger.info("Generating feedback...")
        feedback = feedback_generator.generate_feedback(user_analysis, ref_analysis)
        
        # Create visualization
        logger.info("Creating visualization...")
        visualizer.visualize_comparison(
            user_analysis,
            ref_analysis,
            user_poses,
            ref_poses,
            comparison_output
        )
        
        # Print analysis results
        logger.info("\nAnalysis Results:")
        
        # Print jab count
        logger.info(f"User jabs detected: {len(user_analysis['jabs'])}")
        logger.info(f"Reference jabs detected: {len(ref_analysis['jabs'])}")
        
        # Print feedback
        logger.info("\nFeedback:")
        for point in feedback:
            logger.info(f"  - {point}")
            
        # Print similarity score if available
        if user_analysis['reference_comparison'] and user_analysis['reference_comparison']['average_similarity'] > 0:
            similarity = user_analysis['reference_comparison']['average_similarity'] * 100
            logger.info(f"\nOverall similarity: {similarity:.1f}%")
    else:
        logger.warning(f"User video not found: {user_video_path}")
        logger.info("To analyze your technique, record a video of yourself throwing jabs")
        logger.info("and save it as 'assets/The Wrong Jab.mp4'")
    
    # Save the reference model for future use
    model_path = generate_unique_filename("models", "jab_reference", ".pkl")
    reference_manager.save_references_to_file(model_path)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Print detailed timing summary
    print_timing_summary(total_time)
    
    # Print output file information
    logger.info("\n=== Output Files ===")
    logger.info(f"Reference video analysis: {reference_output}")
    if user_video_exists:
        logger.info(f"User video analysis: {user_output}")
        logger.info(f"Technique comparison: {comparison_output}")
    logger.info(f"Reference model: {model_path}")
    
    logger.info("\nProcessing complete! You can now review the analysis results in the output folder.")

if __name__ == "__main__":
    main()
