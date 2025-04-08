# ShadowCoach: Boxing Form Analyzer

A computer vision-based boxing form analysis application that uses AI to analyze boxing techniques, compare them to reference techniques, and provide detailed feedback.

![ShadowCoach Boxing Analyzer](https://via.placeholder.com/800x400?text=ShadowCoach+Boxing+Analyzer)

## Overview

ShadowCoach Boxing Analyzer is part of the larger ShadowCoach fitness platform. This module specifically focuses on analyzing boxing techniques using computer vision and pose estimation. The application analyzes jab techniques, compares them to reference techniques, and provides detailed feedback on form, speed, and extension.

## Features

- **Real-time Pose Detection**: Uses MediaPipe to detect and track body landmarks
- **Jab Analysis**: Analyzes jab technique including speed, extension, and duration
- **Reference Comparison**: Compares user technique to reference techniques
- **Dynamic Time Warping**: Handles different speeds and orientations of techniques
- **Detailed Feedback**: Provides specific and actionable feedback on technique
- **Visualization**: Creates visualizations comparing user technique to reference
- **API Integration**: FastAPI interface for integration with web applications

## Technologies Used

- **Python**: Core programming language
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Pose estimation and landmark detection
- **NumPy**: Numerical computations
- **SciPy**: Signal processing and dynamic time warping
- **Matplotlib**: Visualization of comparison results
- **FastAPI**: Web API for integration with frontend applications

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
   pip install -r requirements_updated.txt
   ```

4. Prepare video files:
   - Place your reference video in the `assets` folder (e.g., "How to Throw a Jab.mp4")
   - Place your user video in the `assets` folder (e.g., "The Wrong Jab.mp4")
   - Or update the filenames in `new_main.py`

## Usage

### Running the Command-Line Application

Run the main script:
```
python new_main.py
```

### Running the API Server

Run the API server:
```
python run_api.py
```

The API will be available at http://localhost:8000

API Documentation will be available at http://localhost:8000/docs

## Project Structure

The project has been modularized for better maintainability, scalability, and extensibility:

```
shadowcoach-boxer/
├── assets/                  # Video files
├── models/                  # Saved reference models
├── output/                  # Output files (videos, visualizations)
├── shadowcoach/             # Main package
│   ├── __init__.py          # Package initialization
│   ├── analyzers/           # Boxing technique analyzers
│   │   ├── __init__.py
│   │   ├── jab_analyzer.py  # Jab technique analyzer
│   │   └── feedback_generator.py  # Feedback generation
│   ├── api/                 # API integration
│   │   ├── __init__.py
│   │   └── app.py           # FastAPI application
│   ├── core/                # Core functionality
│   │   ├── __init__.py
│   │   ├── pose_processor.py  # Video processing and pose detection
│   │   └── reference_manager.py  # Reference technique management
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py    # File operations
│   │   └── logging_utils.py  # Logging and timing
│   └── visualization/       # Visualization tools
│       ├── __init__.py
│       └── comparison_visualizer.py  # Technique comparison visualization
├── main_old.py              # Original monolithic implementation
├── new_main.py              # New modular implementation
├── run_api.py               # Script to run the API server
├── requirements.txt         # Original dependencies
├── requirements_updated.txt  # Updated dependencies with FastAPI
└── README.md                # Project documentation
```

## How It Works

1. **Video Processing**: Processes frames from a video source using OpenCV
2. **Pose Detection**: Identifies key body landmarks using MediaPipe
3. **Feature Extraction**: Extracts orientation-invariant features from pose data
4. **Jab Detection**: Identifies jabs based on arm movement patterns
5. **Reference Comparison**: Compares user jabs to reference jabs using dynamic time warping
6. **Feedback Generation**: Provides detailed feedback based on the comparison
7. **Visualization**: Creates visualizations comparing user technique to reference

## API Endpoints

- `GET /`: Welcome message
- `GET /references`: List available reference models
- `POST /analyze/jab`: Analyze a jab technique from a video
- `POST /references/upload`: Upload a new reference video

## Future Enhancements

- Support for additional boxing techniques (hooks, crosses, combos, etc.)
- Machine learning for more accurate technique analysis
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

*ShadowCoach Boxing Analyzer is a learning project developed to explore computer vision and pose estimation technologies for boxing technique analysis.*
