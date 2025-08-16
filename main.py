from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from sklearn.cluster import KMeans
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    video_frames=read_video('input_vid/08fd33_4.mp4')
    
    tracker_instance = Tracker('models/last.pt')

    tracks = tracker_instance.get_object_tracks(video_frames,
                                                read_from_stub=True,
                                                stub_path='stubs/track_stubs.pkl')
    
    # get object positions
    tracker_instance.add_position_to_tracks(tracks)
    
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # view tranformer
    view_transformer = ViewTransformer()
    view_transformer.add_transform_position_to_tracks(tracks)
    
    # interpolate ball positions
    tracks["ball"] = tracker_instance.interpolate_ball_positions(tracks["ball"])

    # # save cropped image of a player
    for track_id, player in tracks['players'][0].items():
        bbox=player['bbox']
        frame=video_frames[0]

        # crop bbox from frame
        cropped_image=frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # save the cropped image
        cv2.imwrite(f'output_vid/cropped_image.jpg', cropped_image)
        break
    
    # speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team=team_assigner.get_player_team(video_frames[frame_num],
                                                track['bbox'],
                                                player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    # assign ball aquisition
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

    # Draw and save output video frame by frame
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_vid/output_video.avi', fourcc, 24, (video_frames[0].shape[1], video_frames[0].shape[0]))

    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()
        frame = tracker_instance.draw_annotations_single(frame, tracks, team_ball_control, frame_num)
        frame = camera_movement_estimator.draw_camera_movement_single(frame, camera_movement_per_frame[frame_num])
        frame = speed_and_distance_estimator.draw_speed_and_distance_single(frame, tracks, frame_num)
        out.write(frame)

    out.release()


if __name__ == "__main__":
    main()