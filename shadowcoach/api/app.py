"""
FastAPI application for the ShadowCoach Boxing Analysis System.

This module provides the REST API interface for the ShadowCoach system, allowing
external applications to analyze boxing techniques, manage references, and get feedback.

Key Features:
    - Video upload and processing
    - Jab technique analysis
    - Reference technique management
    - Analysis visualization
    - Detailed feedback generation

Endpoints:
    GET /: Welcome message
    GET /references: List available reference models
    POST /analyze/jab: Analyze a jab technique from video
    POST /references/upload: Upload a new reference video

Dependencies:
    - FastAPI for API framework
    - Core components from shadowcoach.core
    - Analysis components from shadowcoach.analyzers
    - Visualization from shadowcoach.visualization
"""
import os
import shutil
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import core components
from ..core.pose_processor import PoseProcessor
from ..core.reference_manager import ReferenceManager

# Import analyzers
from ..analyzers.jab_analyzer import JabAnalyzer
from ..analyzers.feedback_generator import FeedbackGenerator

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
jab_analyzer = JabAnalyzer(reference_manager)
feedback_generator = FeedbackGenerator()
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
    reference_name: str,
    video: UploadFile = File(...)
):
    """
    Analyze a jab technique from uploaded video.

    This endpoint processes an uploaded video, analyzes the jab technique,
    compares it to a reference, and provides detailed feedback.

    Args:
        background_tasks: FastAPI background tasks handler
        reference_name: Name of the reference technique to compare against
        video: Uploaded video file containing jab technique

    Returns:
        AnalysisResult containing:
            - Number of jabs detected
            - Feedback points
            - Similarity score
            - Link to comparison visualization

    Raises:
        HTTPException(404): If reference_name not found
        HTTPException(500): If analysis fails
    """
    # Check if reference exists
    if reference_name not in reference_manager.list_references():
        raise HTTPException(status_code=404, detail=f"Reference '{reference_name}' not found")

    # Create temporary file for the uploaded video
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "user_video.mp4")

    try:
        # Save uploaded video to temporary file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Create output paths
        user_output = generate_unique_filename("output", "user_analyzed", ".mp4")
        comparison_output = generate_unique_filename("output", "jab_comparison", ".png")

        # Process user video
        logger.info(f"Processing user video: {video.filename}")
        user_poses = pose_processor.process_video(
            temp_video_path,
            output_path=user_output
        )

        # Get reference poses
        reference = reference_manager.get_reference(reference_name)
        if not reference:
            raise HTTPException(status_code=500, detail="Reference data not found")

        ref_poses = reference['poses']

        # Analyze user jabs
        logger.info("Analyzing user jabs...")
        user_analysis = jab_analyzer.analyze_jab(user_poses, reference_name)

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

        # Calculate similarity score if available
        similarity_score = None
        if user_analysis['reference_comparison'] and user_analysis['reference_comparison']['average_similarity'] > 0:
            similarity_score = user_analysis['reference_comparison']['average_similarity'] * 100

        # Clean up temporary files in the background
        background_tasks.add_task(shutil.rmtree, temp_dir)

        # Return analysis results
        return AnalysisResult(
            user_jabs_detected=len(user_analysis['jabs']),
            reference_jabs_detected=len(ref_analysis['jabs']),
            feedback=feedback,
            similarity_score=similarity_score,
            comparison_image_url=f"/output/{os.path.basename(comparison_output)}"
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
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Upload a new reference video

    Args:
        name: Name for the reference
        video: Reference video file
        metadata: Optional metadata for the reference

    Returns:
        Status message
    """
    # Create temporary file for the uploaded video
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "reference_video.mp4")

    try:
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
