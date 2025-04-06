# ShadowCoach: FormTracker

A computer vision-based workout form tracking application that uses AI to monitor exercise form, count repetitions, and provide real-time feedback.

![ShadowCoach FormTracker](https://via.placeholder.com/800x400?text=ShadowCoach+FormTracker)

## Overview

ShadowCoach FormTracker is part of the larger ShadowCoach fitness platform. This module specifically focuses on tracking pushup form using computer vision and pose estimation. The application analyzes body positioning in real-time, counts repetitions, provides form feedback, and saves workout statistics.

## Features

- **Real-time Pose Detection**: Uses MediaPipe to detect and track body landmarks
- **Form Analysis**: Calculates joint angles to analyze exercise form
- **Rep Counting**: Automatically counts completed repetitions
- **Form Feedback**: Provides real-time feedback on exercise form
- **Progress Tracking**: Visual progress bar to track workout goals
- **Workout Statistics**: Saves workout data for future reference
- **Pause/Resume**: Controls to pause and resume workouts
- **Timer**: Tracks workout duration

## Technologies Used

- **Python**: Core programming language
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Pose estimation and landmark detection
- **NumPy**: Numerical computations and angle calculations
- **JSON**: Data storage for workout history

## Installation

### Prerequisites

- Python 3.6+
- Webcam or video input source

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/ondiekelijah/ShadowCoach-FormTracker.git
   cd ShadowCoach-FormTracker
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare a video file:
   - Place your workout video in the `assets` folder
   - Default video should be named `pushup.mp4`
   - Or update the filename in `main.py`

## Usage

Run the main script:
```
python main.py
```

### Controls

- **Q**: Quit and save workout data
- **R**: Reset counter
- **P**: Pause/resume workout

## Project Structure

- `main.py` - Main application with pose detection and UI rendering
- `requirements.txt` - Project dependencies
- `workouts.json` - Saved workout data
- `assets/` - Folder for video files

## How It Works

1. **Video Input**: Processes frames from a video source
2. **Pose Detection**: Identifies key body landmarks using MediaPipe
3. **Angle Calculation**: Calculates joint angles between landmarks
4. **Stage Detection**: Determines exercise stage (up/down) based on angles
5. **Rep Counting**: Counts completed repetitions based on stage transitions
6. **Form Analysis**: Provides feedback based on joint angles
7. **Data Storage**: Saves workout statistics to JSON file

## Future Enhancements

- Support for additional exercises (squats, lunges, etc.)
- Machine learning for personalized form feedback
- User profiles and authentication
- Mobile app integration
- Social features and challenges
- Comprehensive analytics dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the pose estimation framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
- All contributors and supporters of the project

---

*ShadowCoach FormTracker is a learning project developed to explore computer vision and pose estimation technologies for fitness applications.*
