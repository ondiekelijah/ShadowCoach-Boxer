#!/usr/bin/env python3
"""
ShadowCoach Boxing Analysis System - Main Entry Point
A modular system for analyzing boxing techniques using computer vision
"""
import time
import os
import argparse
from pathlib import Path

# Import core components
from shadowcoach.core.pose_processor import PoseProcessor
from shadowcoach.core.reference_manager import ReferenceManager

# Import analyzers
from shadowcoach.analyzers.enhanced_jab_analyzer import EnhancedJabAnalyzer
from shadowcoach.analyzers.enhanced_feedback_generator import EnhancedFeedbackGenerator

# Import visualization
from shadowcoach.visualization.comparison_visualizer import ComparisonVisualizer

# Import utilities
from shadowcoach.utils.logging_utils import logger, print_timing_summary
from shadowcoach.utils.file_utils import generate_unique_filename, ensure_directory_exists

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ShadowCoach Boxing Analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to the video to analyze")
    parser.add_argument("--references", type=str, required=True, help="Comma-separated list of reference names to compare against")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    # Initialize components
    pose_processor = PoseProcessor()
    reference_manager = ReferenceManager()
    jab_analyzer = EnhancedJabAnalyzer(reference_manager)
    feedback_generator = EnhancedFeedbackGenerator()
    visualizer = ComparisonVisualizer()
    
    # Define video paths
    video_path = args.video
    
    # Parse reference names
    reference_list = [name.strip() for name in args.references.split(',')]
    
    # Check if references exist
    for ref_name in reference_list:
        if ref_name not in reference_manager.list_references():
            logger.error(f"Reference '{ref_name}' not found")
            return
    
    # Process video
    logger.info(f"Processing video: {video_path}")
    pose_data = pose_processor.process_video(video_path)
    
    # Analyze with primary reference (first in the list)
    primary_ref = reference_list[0]
    logger.info(f"Analyzing jabs with primary reference: {primary_ref}")
    primary_analysis = jab_analyzer.analyze_jab(pose_data, primary_ref)
    
    # If multiple references provided, analyze with each additional reference
    additional_analyses = {}
    if len(reference_list) > 1:
        for ref_name in reference_list[1:]:
            logger.info(f"Analyzing jabs with additional reference: {ref_name}")
            additional_analyses[ref_name] = jab_analyzer.analyze_jab(pose_data, ref_name)
    
    # Get reference analyses
    reference_analyses = {}
    for ref_name in reference_list:
        logger.info(f"Getting analysis for reference: {ref_name}")
        reference_analyses[ref_name] = jab_analyzer.get_reference_analysis(ref_name)
    
    # Generate feedback based on primary reference
    logger.info("Generating feedback...")
    if len(reference_list) > 1:
        # Generate comparative feedback across all references
        feedback = feedback_generator.generate_comparative_feedback(
            primary_analysis,
            reference_analyses
        )
    else:
        # Generate standard feedback for single reference
        feedback = feedback_generator.generate_feedback(
            primary_analysis, 
            reference_analyses[primary_ref]
        )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    os.makedirs(args.output, exist_ok=True)
    
    # Primary reference visualization
    primary_output = os.path.join(args.output, f"jab_comparison_{primary_ref}.png")
    visualizer.visualize_comparison(
        primary_analysis, 
        reference_analyses[primary_ref], 
        pose_data, 
        reference_manager.get_reference(primary_ref)['poses'],
        primary_output
    )
    
    # Additional reference visualizations
    if len(reference_list) > 1:
        for ref_name in reference_list[1:]:
            additional_output = os.path.join(args.output, f"jab_comparison_{ref_name}.png")
            visualizer.visualize_comparison(
                additional_analyses[ref_name], 
                reference_analyses[ref_name], 
                pose_data, 
                reference_manager.get_reference(ref_name)['poses'],
                additional_output
            )
    
    # Calculate similarity scores
    similarity_scores = {}
    similarity_scores[primary_ref] = None
    if primary_analysis['reference_comparison'] and primary_analysis['reference_comparison']['average_similarity'] > 0:
        similarity_scores[primary_ref] = primary_analysis['reference_comparison']['average_similarity'] * 100
    
    if len(reference_list) > 1:
        for ref_name in reference_list[1:]:
            similarity_scores[ref_name] = None
            if additional_analyses[ref_name]['reference_comparison'] and additional_analyses[ref_name]['reference_comparison']['average_similarity'] > 0:
                similarity_scores[ref_name] = additional_analyses[ref_name]['reference_comparison']['average_similarity'] * 100
    
    # Print analysis results
    logger.info("\nAnalysis Results:")
    
    # Print jab count
    logger.info(f"User jabs detected: {len(primary_analysis['jabs'])}")
    for ref_name in reference_list:
        logger.info(f"Reference '{ref_name}' jabs detected: {len(reference_analyses[ref_name]['jabs'])}")
    
    # Print similarity scores
    logger.info("\nSimilarity Scores:")
    for ref_name, score in similarity_scores.items():
        if score is not None:
            logger.info(f"  - Similarity to '{ref_name}': {score:.1f}%")
    
    # Print technical scores if available
    if 'technical_scores' in primary_analysis:
        logger.info("\nTechnical Scores:")
        for element, score in primary_analysis['technical_scores'].items():
            # Format the element name for better readability
            element_name = element.replace('_', ' ').title()
            logger.info(f"  - {element_name}: {score:.1f}%")
    
    # Print common errors if available
    if 'enhanced_features' in primary_analysis and primary_analysis['enhanced_features']:
        # Aggregate errors across all jabs
        error_types = ['telegraphing', 'looping', 'pawing', 'overextending', 'dropping_guard', 'poor_retraction']
        aggregated_errors = {error: False for error in error_types}
        
        # Set an error to True if it's detected in any jab
        for jab_feature in primary_analysis['enhanced_features']:
            if jab_feature.get('common_errors'):
                for error, detected in jab_feature['common_errors'].items():
                    if detected:
                        aggregated_errors[error] = True
        
        # Display detected errors
        detected_errors = [error for error, detected in aggregated_errors.items() if detected]
        if detected_errors:
            logger.info("\nDetected Errors:")
            for error in detected_errors:
                # Format the error name for better readability
                error_name = error.replace('_', ' ').title()
                logger.info(f"  - {error_name}")
    
    # Print feedback
    logger.info("\nFeedback:")
    for point in feedback:
        logger.info(f"  - {point}")
    
    logger.info(f"\nVisualizations saved to {args.output}")

if __name__ == "__main__":
    main()
