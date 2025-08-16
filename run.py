import cv2
import numpy as np
import json
import os
from utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator

def auto_calibrate_view_transformer(video_path, output_config_path=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read video file")
        return None
    detector = AutoFieldDetector()
    corners = detector.detect_field_corners(frame)
    if corners is not None:
        if output_config_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_config_path = f'config_{video_name}.json'
        config = {
            'pixel_vertices': corners.tolist(),
            'video_path': video_path,
            'detection_method': 'automatic'
        }
        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Detected corners: {corners.tolist()}")
        print(f"Configuration saved to {output_config_path}")
        return output_config_path
    else:
        print("Automatic detection failed")
        return None

class AutoFieldDetector:
    def __init__(self):
        self.field_color_lower = np.array([35, 50, 50])
        self.field_color_upper = np.array([85, 255, 255])
    def detect_field_corners(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.field_color_lower, self.field_color_upper)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            return self.order_corners(corners)
        return None
    def order_corners(self, corners):
        top_corners = corners[corners[:, 1].argsort()][:2]
        bottom_corners = corners[corners[:, 1].argsort()][2:]
        top_corners = top_corners[top_corners[:, 0].argsort()]
        bottom_corners = bottom_corners[bottom_corners[:, 0].argsort()]
        return np.array([
            bottom_corners[0],
            top_corners[0],
            top_corners[1],
            bottom_corners[1]
        ])

def main(video_path, config_path=None, output_name=None):
    print(f"Processing video: {video_path}")
    video_frames = read_video(video_path)
    print(f"Loaded {len(video_frames)} frames")
    tracker_instance = Tracker('models/last.pt')
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    track_stub_path = f'stubs/track_stubs_{video_name}.pkl'
    camera_stub_path = f'stubs/camera_movement_stub_{video_name}.pkl'
    tracks = tracker_instance.get_object_tracks(video_frames, read_from_stub=True, stub_path=track_stub_path)
    tracker_instance.add_position_to_tracks(tracks)
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path=camera_stub_path)
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    view_transformer = ViewTransformer(config_path)
    view_transformer.add_transform_position_to_tracks(tracks)
    tracks["ball"] = tracker_instance.interpolate_ball_positions(tracks["ball"])
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) == 0:
                team_ball_control.append(0)
            else:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if output_name is None:
        output_name = f'output_video_{video_name}.avi'
    elif not output_name.endswith('.avi'):
        output_name += '.avi'
    output_dir = 'output_vid'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_name)
    out = cv2.VideoWriter(output_path, fourcc, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))
    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()
        frame = tracker_instance.draw_annotations_single(frame, tracks, team_ball_control, frame_num)
        frame = camera_movement_estimator.draw_camera_movement_single(frame, camera_movement_per_frame[frame_num])
        frame = speed_and_distance_estimator.draw_speed_and_distance_single(frame, tracks, frame_num)
        out.write(frame)
    out.release()
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    video_path = input("Enter path to input video: ").strip()
    if not os.path.exists(video_path):
        print("Video file does not exist")
        exit(1)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_config_path = f'config_{video_name}.json'
    print(f"Generating configuration file: {output_config_path}")
    config_path = auto_calibrate_view_transformer(video_path, output_config_path)
    if config_path is None:
        print("Field detection failed")
        exit(1)
    output_name = input("Enter custom output filename (without .avi extension, press Enter to use default): ").strip()
    if output_name == '':
        output_name = None
    main(video_path, config_path, output_name)
