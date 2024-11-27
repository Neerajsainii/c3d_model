import cv2
import os
import numpy as np

def video_to_frames(video_path, frame_count=16, resize_dim=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize_dim)
        frame = frame.astype(np.float32) / 255.0  # Normalize
        frames.append(frame)
    
    frames = np.array(frames)
    return frames

def load_videos_from_directory(directory, frame_count=16, resize_dim=(112, 112)):
    video_data = []
    labels = []
    
    for label, video_folder in enumerate(os.listdir(directory)):
        video_folder_path = os.path.join(directory, video_folder)
        # print(video_folder_path)
        
        if os.path.isdir(video_folder_path):
            for video_file in os.listdir(video_folder_path):
                # print(video_file)
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(video_folder_path, video_file)
                    frames = video_to_frames(video_path, frame_count, resize_dim)
                    video_data.append(frames)
                    labels.append(label)  # 0 for Normal, 1 for Anomaly
    
    return np.array(video_data), np.array(labels)

# Example usage:
directory = "../data/UCF"  # Your dataset directory
# print(os.listdir(directory))
X, y = load_videos_from_directory(directory)
