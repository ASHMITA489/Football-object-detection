# Football object detection

This provides a complete pipeline for analyzing the football match videos. It detects and tracks players, referees, and the ball, estimates camera movement, assigns teams, calculates player speeds and distances, and generates annotated output videos.


## Project structure

- camera_movement_estimator/                                         # Camera movement estimation logic
- input_vid/                                                         # Place your input videos here
- main.py                                                            # Script for custom video
- models/                                                            # YOLO model weights
- output_vid/                                                        # Output videos and images
- player_ball_assigner/                                              # Ball possession assignment logic
- requirements.txt                                                   # Python dependencies
- run.py                                                             # Main script for giving input video
- speed_and_distance_estimator/                                      # Speed and distance estimation logic
- stubs/                                                             # Cached detection/camera movement results
- team_assigner/                                                     # Team color clustering and assignment
- trackers/                                                          # Object detection and tracking
- training/                                                          # Notebooks and data for model training
- utils/                                                             # Utility functions (video, bbox, etc.)
- view_transformer/                                                  # Field calibration and perspective transform


## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Football-object-detection
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```
3. Run the pipeline
   ```bash
   python run.py
   ```
4. Enter the input video file

