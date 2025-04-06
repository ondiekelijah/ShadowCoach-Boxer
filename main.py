import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import datetime

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED =(0, 0, 255)
BLUE = (255, 0, 0)

def process_frame(frame, pose):
    """
    Process a video frame for pose detection.
    
    Args:
        frame (numpy.ndarray): Input video frame
        pose (mediapipe.python.solutions.pose.Pose): MediaPipe pose detection model
        
    Returns:
        tuple: Processed image, pose detection results, frame height, and frame width
    """
    # extract dimension of the frame
    height, width, _ = frame.shape
    
    # convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # process frame with pose from mediapipe
    results = pose.process(image)
    
    # convert back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results, height, width

def extract_landmarks(results, mp_pose):
    """
    Extract key landmarks (shoulder, elbow, wrist) from pose detection results.
    
    Args:
        results (mediapipe.python.solutions.pose.PoseLandmarkList): Pose detection results
        mp_pose (mediapipe.python.solutions.pose): MediaPipe pose module
        
    Returns:
        tuple: Coordinates of shoulder, elbow, and wrist landmarks, or None values if landmarks not detected
    """
    try:
        landmarks = results.pose_landmarks.landmark
        
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        return shoulder, elbow, wrist

    except AttributeError:
        return None, None, None


def calculate_angle(shoulder, elbow, wrist):
    """
    Calculate the angle between three points (shoulder, elbow, wrist).
    
    Args:
        shoulder (list): Coordinates [x, y] of the shoulder point
        elbow (list): Coordinates [x, y] of the elbow point
        wrist (list): Coordinates [x, y] of the wrist point
        
    Returns:
        float: Angle in degrees between the three points
    """
    # Calculate vectors between points
    a = np.array(shoulder) - np.array(elbow)
    b = np.array(wrist) - np.array(elbow)
    
    # Calculate the angle using the dot product formula
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Handle potential numerical errors to ensure the value is within [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate the angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def get_form_feedback(angle):
    """
    Generate form feedback based on the measured angle.
    
    Args:
        angle (float or None): The measured angle between shoulder, elbow, and wrist
        
    Returns:
        str or None: Feedback message based on the angle, or None if angle is None
    """
    if angle is None:
        return None
    
    if angle < 40:
        return "Too low! Raise your body"
    elif angle < 70:
        return "Good depth!"
    elif angle > 160:
        return "Full extension!"
    elif angle > 140:
        return "Good form!"
    else:
        return "Keep going!"

def render_ui(image, angle, width, stage=None, counter=0, elapsed_time=0, feedback=None, target_reps=10):
    """
    Render the user interface elements on the image.
    
    Args:
        image (numpy.ndarray): Input image to draw UI elements on
        angle (float or None): The measured angle between shoulder, elbow, and wrist
        width (int): Width of the image
        stage (str, optional): Current exercise stage (Up/Down). Defaults to None.
        counter (int, optional): Number of completed repetitions. Defaults to 0.
        elapsed_time (float, optional): Elapsed workout time in seconds. Defaults to 0.
        feedback (str, optional): Form feedback message. Defaults to None.
        target_reps (int, optional): Target number of repetitions. Defaults to 10.
        
    Returns:
        numpy.ndarray: Image with UI elements drawn on it
    """
    angle_max = 180  # Max angle when arm is fully extended
    angle_min = 25   # Min angle when arm is bent

    # Create a copy of the image to avoid modifying the original
    output_image = image.copy()
    
    # Draw rectangle for angle display
    cv2.rectangle(output_image, (int(width/2) - 150, 0), (int(width/2) + 150, 100), BLUE, -1)
    
    # Display the angle value with color feedback based on range
    angle_color = WHITE
    if angle is not None:
        # Change color based on angle range
        if angle < angle_min:
            angle_color = RED  # Too bent
        elif angle > angle_max - 30:
            angle_color = GREEN  # Good form
        
        angle_text = f"Angle: {int(angle)}"
    else:
        angle_text = "No angle detected"
    
    cv2.putText(output_image, angle_text, (int(width/2) - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, angle_color, 2)
    
    # Add title at the top
    cv2.putText(output_image, "AI Workout Manager", (int(width/2) - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
    
    # Display stage (up/down)
    if stage:
        cv2.rectangle(output_image, (0, 0), (200, 60), BLUE, -1)
        cv2.putText(output_image, f"Stage: {stage}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    
    # Display pushup counter with progress bar
    progress_width = 300
    cv2.rectangle(output_image, (0, 70), (200, 130), BLUE, -1)
    cv2.putText(output_image, f"Count: {counter}/{target_reps}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    
    # Draw progress bar
    progress_percent = min(counter / target_reps, 1.0) if target_reps > 0 else 0
    cv2.rectangle(output_image, (210, 85), (210 + progress_width, 115), BLACK, -1)
    cv2.rectangle(output_image, (210, 85), (210 + int(progress_width * progress_percent), 115), GREEN, -1)
    
    # Display timer
    minutes, seconds = divmod(int(elapsed_time), 60)
    cv2.rectangle(output_image, (0, 140), (200, 200), BLUE, -1)
    cv2.putText(output_image, f"Time: {minutes:02d}:{seconds:02d}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    
    # Display form feedback
    if feedback:
        cv2.rectangle(output_image, (width - 400, 0), (width, 60), BLUE, -1)
        cv2.putText(output_image, feedback, (width - 390, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    
    # Add keyboard shortcuts guide
    h, w, _ = image.shape
    shortcut_y_start = h - 160
    cv2.rectangle(output_image, (width - 220, shortcut_y_start), (width, shortcut_y_start + 150), BLACK, -1)
    cv2.rectangle(output_image, (width - 220, shortcut_y_start), (width, shortcut_y_start + 30), BLUE, -1)
    cv2.putText(output_image, "CONTROLS", (width - 210, shortcut_y_start + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    
    # Add individual shortcuts
    shortcuts = [
        ("Q", "Quit & Save"),
        ("R", "Reset Counter"),
        ("P", "Pause/Resume")
    ]
    
    for i, (key, action) in enumerate(shortcuts):
        y_pos = shortcut_y_start + 60 + (i * 30)
        cv2.putText(output_image, f"{key}:", (width - 210, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        cv2.putText(output_image, action, (width - 170, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1)
    
    return output_image

def process_video_frame(frame, pose, mp_pose, mp_drawing, stage_data, elapsed_time, target_reps=10):
    """
    Process a single video frame for pose detection and UI rendering.
    
    Args:
        frame (numpy.ndarray): Input video frame
        pose (mediapipe.python.solutions.pose.Pose): MediaPipe pose detection model
        mp_pose (mediapipe.python.solutions.pose): MediaPipe pose module
        mp_drawing (mediapipe.python.solutions.drawing_utils): MediaPipe drawing utilities
        stage_data (dict): Dictionary containing current stage and counter information
        elapsed_time (float): Elapsed workout time in seconds
        target_reps (int, optional): Target number of repetitions. Defaults to 10.
        
    Returns:
        numpy.ndarray: Processed image with pose landmarks and UI elements
    """
    image, results, _, width = process_frame(frame, pose)
    
    # Extract landmarks and calculate angle
    shoulder, elbow, wrist = extract_landmarks(results, mp_pose)
    
    angle = None
    feedback = None
    if shoulder is not None and elbow is not None and wrist is not None:
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # Get form feedback based on angle
        feedback = get_form_feedback(angle)
        
        # Determine stage based on angle
        if angle is not None:
            if angle > 160:
                current_stage = "Up"
            elif angle < 80:
                current_stage = "Down"
            else:
                current_stage = stage_data["current_stage"]
                
            # Count a pushup when transitioning from down to up
            if stage_data["current_stage"] == "Down" and current_stage == "Up":
                stage_data["counter"] += 1
                
            stage_data["current_stage"] = current_stage
    
    # Render UI elements
    image = render_ui(image, angle, width, stage_data["current_stage"], 
                     stage_data["counter"], elapsed_time, feedback, target_reps)
    
    # Draw landmarks if they exist
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245,117,16), thickness=2, circle_radius=2), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    return image

def save_workout_data(exercise_type, reps, duration, filename="workouts.json"):
    """
    Save workout data to a JSON file.
    
    Args:
        exercise_type (str): Type of exercise performed
        reps (int): Number of repetitions completed
        duration (float): Duration of the workout in seconds
        filename (str, optional): Path to the JSON file. Defaults to "workouts.json".
        
    Returns:
        dict: Workout data that was saved
    """
    # Create workout data entry
    workout_data = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exercise": exercise_type,
        "reps": reps,
        "duration": round(duration, 2),  # Round to 2 decimal places
        "avg_reps_per_min": round(reps / (duration / 60), 2) if duration > 0 else 0
    }
    
    # Load existing data if file exists
    all_workouts = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                all_workouts = json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted, start with empty list
            all_workouts = []
    
    # Append new workout
    all_workouts.append(workout_data)
    
    # Save back to file
    with open(filename, 'w') as f:
        json.dump(all_workouts, f, indent=4)
    
    print(f"Workout saved to {filename}")
    return workout_data

def handle_key_events(key, stage_data, is_paused, pause_start_time, paused_time):
    """
    Handle keyboard events during workout.
    
    Args:
        key (int): Key code from cv2.waitKey
        stage_data (dict): Dictionary containing current stage and counter information
        is_paused (bool): Whether the workout is currently paused
        pause_start_time (float): Time when the workout was paused
        paused_time (float): Total time the workout has been paused
        
    Returns:
        tuple: Updated is_paused state, paused_time, and whether to exit the loop
    """
    exit_loop = False
    
    if key == ord('q'):
        exit_loop = True
    elif key == ord('r'):
        # Reset counter
        stage_data["counter"] = 0
        print("Counter reset to 0")
    elif key == ord('p'):
        # Toggle pause
        if is_paused:
            # Resume - update the paused time
            paused_time += time.time() - pause_start_time
            print("Workout resumed")
        else:
            # Pause - record when we paused
            pause_start_time = time.time()
            print("Workout paused")
        is_paused = not is_paused
    
    return is_paused, paused_time, exit_loop, pause_start_time

def run_pose_detection(mp_drawing, mp_pose, filename, target_reps=10):
    """
    Main function to run pose detection on a video file.
    
    Args:
        mp_drawing (mediapipe.python.solutions.drawing_utils): MediaPipe drawing utilities
        mp_pose (mediapipe.python.solutions.pose): MediaPipe pose module
        filename (str): Path to the video file
        target_reps (int, optional): Target number of repetitions. Defaults to 10.
        
    Returns:
        int: Number of completed repetitions
    """
    cap = cv2.VideoCapture(filename)
    
    # Initialize stage data
    stage_data = {
        "current_stage": "None",
        "counter": 0
    }
    
    # Initialize timer
    start_time = time.time()
    paused_time = 0
    is_paused = False
    pause_start_time = 0
    
    # Initialize exercise type
    exercise_type = "Pushup"
    
    # Display instructions
    print("Controls:")
    print("  'q' - Quit and save workout data")
    print("  'r' - Reset counter")
    print("  'p' - Pause/resume workout")
    print(f"  Target: {target_reps} reps")
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Check if we've reached the end of the video
            if not ret:
                break
            
            # Calculate elapsed time
            if not is_paused:
                elapsed_time = time.time() - start_time - paused_time
            
            # Process the frame
            image = process_video_frame(frame, pose, mp_pose, mp_drawing, stage_data, elapsed_time, target_reps)
            
            # Add pause indicator if paused
            if is_paused:
                h, w, _ = frame.shape
                cv2.putText(image, "PAUSED", (int(w/2) - 100, int(h/2)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 3)
            
            # Check if target reached
            if stage_data["counter"] >= target_reps:
                h, w, _ = frame.shape
                cv2.putText(image, "TARGET REACHED!", (int(w/2) - 200, int(h/2) - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)
            
            # Display the result
            cv2.imshow("AI Workout Manager", image)
            
            # Handle key presses
            key = cv2.waitKey(10) & 0xFF
            is_paused, paused_time, exit_loop, pause_start_time = handle_key_events(
                key, stage_data, is_paused, pause_start_time, paused_time
            )
            
            if exit_loop:
                break
    
    # Calculate final workout duration
    total_duration = time.time() - start_time - paused_time
    
    # Save workout data
    workout_summary = save_workout_data(exercise_type, stage_data["counter"], total_duration)
    
    # Display workout summary
    print("\nWorkout Summary:")
    print(f"Exercise: {workout_summary['exercise']}")
    print(f"Reps completed: {workout_summary['reps']} / {target_reps} target")
    print(f"Duration: {workout_summary['duration']} seconds")
    print(f"Average reps per minute: {workout_summary['avg_reps_per_min']}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Return the final count
    return stage_data["counter"]

if __name__ == "__main__":
    """
    Main entry point of the application.
    Initializes MediaPipe components and runs the pose detection.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Set target reps (can be changed by the user)
    target_reps = 10
    
    # Run pose detection
    run_pose_detection(mp_drawing, mp_pose, "assets/pushup.mp4", target_reps)
