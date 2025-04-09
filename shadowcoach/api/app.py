"""
FastAPI application for the ShadowCoach Boxing Analysis System
"""
import os
import shutil
import tempfile
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import logging

# Import core components
from ..core.pose_processor import PoseProcessor
from ..core.reference_manager import ReferenceManager

# Import analyzers
from ..analyzers.jab_analyzer import JabAnalyzer
from ..analyzers.enhanced_jab_analyzer import EnhancedJabAnalyzer
from ..analyzers.feedback_generator import FeedbackGenerator
from ..analyzers.enhanced_feedback_generator import EnhancedFeedbackGenerator

# Import visualization
from ..visualization.comparison_visualizer import ComparisonVisualizer

# Import utilities
from ..utils.logging_utils import logger
from ..utils.file_utils import generate_unique_filename, ensure_directory_exists

# Create FastAPI app
app = FastAPI(
    title="ShadowCoach API",
    description="API for analyzing boxing techniques",
    version="0.1.0"
)

# Initialize components
pose_processor = PoseProcessor()
reference_manager = ReferenceManager()
jab_analyzer = EnhancedJabAnalyzer(reference_manager)  # Using enhanced analyzer
feedback_generator = EnhancedFeedbackGenerator()  # Using enhanced feedback generator
visualizer = ComparisonVisualizer()

# Create output directories
ensure_directory_exists("output")
ensure_directory_exists("models")
ensure_directory_exists("uploads")

# Mount static files
app.mount("/output", StaticFiles(directory="output"), name="output")

# Load reference models if available
reference_files = list(Path("models").glob("*.pkl"))
if reference_files:
    # Load the most recent reference file
    latest_reference = max(reference_files, key=os.path.getctime)
    logger.info(f"Loading reference models from {latest_reference}")
    reference_manager.load_references_from_file(str(latest_reference))

# Define response models
class AnalysisResult(BaseModel):
    """Analysis result model"""
    user_jabs_detected: int
    reference_jabs_detected: int
    feedback: List[str]
    similarity_score: Optional[float] = None
    comparison_image_url: Optional[str] = None
    technical_scores: Optional[Dict[str, float]] = None
    common_errors: Optional[Dict[str, bool]] = None

class ReferenceAnalysis(BaseModel):
    """Reference analysis result model"""
    jabs_detected: int
    similarity_score: Optional[float] = None
    comparison_image_url: str

class MultiReferenceAnalysisResult(BaseModel):
    """Multi-reference analysis result model"""
    user_jabs_detected: int
    reference_analyses: Dict[str, ReferenceAnalysis]
    primary_reference: str
    feedback: List[str]
    technical_scores: Optional[Dict[str, float]] = None
    common_errors: Optional[Dict[str, bool]] = None

# Helper functions for API endpoints
def _extract_common_errors(analysis: Dict[str, Any]) -> Dict[str, bool]:
    """
    Extract and aggregate common errors from analysis
    
    Args:
        analysis: Analysis data
        
    Returns:
        Dictionary of common errors
    """
    common_errors = {}
    
    # If no enhanced features, return empty dict
    if not analysis.get('enhanced_features'):
        return common_errors
        
    # Initialize all possible errors as False
    error_types = ['telegraphing', 'looping', 'pawing', 'overextending', 'dropping_guard', 'poor_retraction']
    common_errors = {error: False for error in error_types}
    
    # Set an error to True if it's detected in any jab
    for jab_feature in analysis['enhanced_features']:
        if not jab_feature.get('common_errors'):
            continue
            
        for error, detected in jab_feature['common_errors'].items():
            if detected:
                common_errors[error] = True
    
    return common_errors

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to ShadowCoach API"}

@app.get("/references")
async def get_references():
    """Get available reference models"""
    return {"references": reference_manager.list_references()}

@app.post("/analyze/jab")
async def analyze_jab(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    reference_names: str = Query(...)  # Changed from reference_name to reference_names
):
    """
    Analyze a jab technique from a video comparing against one or multiple references
    
    Args:
        reference_names: Comma-separated names of references to compare against
        video: Video file to analyze
    
    Returns:
        Analysis results
    """
    # Parse reference names (comma-separated)
    reference_list = [name.strip() for name in reference_names.split(',')]
    
    # Check if references exist
    for ref_name in reference_list:
        if ref_name not in reference_manager.list_references():
            raise HTTPException(status_code=404, detail=f"Reference '{ref_name}' not found")
    
    # Create temporary file for the uploaded video
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "user_video.mp4")
    
    try:
        # Save uploaded video to temporary file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process user video
        logger.info(f"Processing user video: {video.filename}")
        user_poses = pose_processor.process_video(temp_video_path)
        
        # Analyze user jabs with primary reference (first in the list)
        logger.info(f"Analyzing jabs with primary reference: {reference_list[0]}")
        primary_analysis = jab_analyzer.analyze_jab(user_poses, reference_list[0])
        
        # If multiple references provided, analyze with each additional reference
        additional_analyses = {}
        if len(reference_list) > 1:
            for ref_name in reference_list[1:]:
                logger.info(f"Analyzing jabs with additional reference: {ref_name}")
                additional_analyses[ref_name] = jab_analyzer.analyze_jab(user_poses, ref_name)
        
        # Get reference analyses
        reference_analyses = {}
        for ref_name in reference_list:
            logger.info(f"Getting analysis for reference: {ref_name}")
            reference_analyses[ref_name] = jab_analyzer.get_reference_analysis(ref_name)
        
        # Generate feedback based on primary reference
        logger.info("Generating feedback...")
        if len(reference_list) > 1:
            # Generate comparative feedback across all references
            primary_feedback = feedback_generator.generate_comparative_feedback(
                primary_analysis,
                reference_analyses
            )
        else:
            # Generate standard feedback for single reference
            primary_feedback = feedback_generator.generate_feedback(
                primary_analysis, 
                reference_analyses[reference_list[0]]
            )
        
        # Create visualization for primary reference
        logger.info("Creating visualization...")
        comparison_outputs = {}
        primary_output = os.path.join("output", f"jab_comparison_{reference_list[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        visualizer.visualize_comparison(
            primary_analysis, 
            reference_analyses[reference_list[0]], 
            user_poses, 
            reference_manager.get_reference(reference_list[0])['poses'],
            primary_output
        )
        comparison_outputs[reference_list[0]] = f"/output/{os.path.basename(primary_output)}"
        
        # Create visualizations for additional references
        if len(reference_list) > 1:
            for ref_name in reference_list[1:]:
                additional_output = os.path.join("output", f"jab_comparison_{ref_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                visualizer.visualize_comparison(
                    additional_analyses[ref_name], 
                    reference_analyses[ref_name], 
                    user_poses, 
                    reference_manager.get_reference(ref_name)['poses'],
                    additional_output
                )
                comparison_outputs[ref_name] = f"/output/{os.path.basename(additional_output)}"
        
        # Calculate similarity scores
        similarity_scores = {}
        similarity_scores[reference_list[0]] = None
        if primary_analysis['reference_comparison'] and primary_analysis['reference_comparison']['average_similarity'] > 0:
            similarity_scores[reference_list[0]] = primary_analysis['reference_comparison']['average_similarity'] * 100
        
        if len(reference_list) > 1:
            for ref_name in reference_list[1:]:
                similarity_scores[ref_name] = None
                if additional_analyses[ref_name]['reference_comparison'] and additional_analyses[ref_name]['reference_comparison']['average_similarity'] > 0:
                    similarity_scores[ref_name] = additional_analyses[ref_name]['reference_comparison']['average_similarity'] * 100
        
        # Extract technical scores from primary analysis
        technical_scores = primary_analysis.get('technical_scores', None)
        
        # Get common errors from primary analysis
        common_errors = _extract_common_errors(primary_analysis)
        
        # Clean up temporary files in the background
        background_tasks.add_task(shutil.rmtree, temp_dir)
        
        # Return analysis results
        return MultiReferenceAnalysisResult(
            user_jabs_detected=len(primary_analysis['jabs']),
            reference_analyses={
                ref_name: ReferenceAnalysis(
                    jabs_detected=len(reference_analyses[ref_name]['jabs']),
                    similarity_score=similarity_scores[ref_name],
                    comparison_image_url=comparison_outputs[ref_name]
                ) for ref_name in reference_list
            },
            primary_reference=reference_list[0],
            feedback=primary_feedback,
            technical_scores=technical_scores,
            common_errors=common_errors
        )
    
    except Exception as e:
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        logger.error(f"Error analyzing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/references/upload")
async def upload_reference(
    name: str,
    video: UploadFile = File(...),
    metadata_str: Optional[str] = None
):
    """
    Upload a new reference video
    
    Args:
        name: Name for the reference
        video: Reference video file
        metadata_str: Optional metadata for the reference as a JSON string
        
    Returns:
        Status message
    """
    # Create temporary file for the uploaded video
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "reference_video.mp4")
    
    try:
        # Parse metadata if provided
        metadata = None
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
                logger.info(f"Parsed metadata: {metadata}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing metadata: {e}")
                metadata = {"error": "Failed to parse metadata"}
        
        # Save uploaded video to temporary file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Create output path
        reference_output = generate_unique_filename("output", "reference_analyzed", ".mp4")
        
        # Process reference video
        logger.info(f"Processing reference video: {video.filename}")
        ref_poses = pose_processor.process_video(
            temp_video_path, 
            output_path=reference_output
        )
        
        # Save reference
        reference_manager.save_reference(name, ref_poses, metadata or {
            "uploaded_filename": video.filename
        })
        
        # Save references to file
        model_path = generate_unique_filename("models", "reference_model", ".pkl")
        reference_manager.save_references_to_file(model_path)
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        return {
            "message": f"Reference '{name}' uploaded successfully",
            "reference_video": f"/output/{os.path.basename(reference_output)}",
            "model_path": model_path
        }
    
    except Exception as e:
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        logger.error(f"Error uploading reference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
