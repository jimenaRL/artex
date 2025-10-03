"""
Extracts high-quality JPEG image from a video using FFmpeg.
"""
import os
import subprocess
from argparse import ArgumentParser

def extract_frame(video_path, frame_number, output_folder):

    output_image = os.path.join(output_folder, f"frame_{frame_number}.png")
    # Prepare the ffmpeg command
    command = [
        'ffmpeg',
        '-i', video_path,                          # Input video file
        '-vf', f'select=eq(n\\,{frame_number})',    # Select the specific frame number
        '-q:v', '2',                               # Set quality level (2 = high quality)
        '-frames:v', '1',                          # Output only one frame
        output_image                               # Output JPEG file
    ]

    try:
        # Run the ffmpeg command
        subprocess.run(command, check=True)
        print(f'Frame {frame_number} extracted successfully to {output_image}')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred: {e}')

def extract_all_frames(video_path, output_folder):

    output_pattern = os.path.join(output_folder, f"%06d.png")
    # Prepare the ffmpeg command
    command = [
        'ffmpeg',
        '-i', video_path,                          # Input video file
        output_pattern                               # Output pattern for files
    ]

    try:
        # Run the ffmpeg command
        print(f"RUNNING: {command}")
        subprocess.run(command, check=True)
        print(f'Frames extracted successfully to {output_folder}')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred: {e}')

if __name__ == "__main__":

    ap = ArgumentParser()
    ap.add_argument(
        '--video', type=str, default="ForLearning_2025-10-01_13-15-24.mp4")

    args = ap.parse_args()
    video_path = args.video

    parent_folder, file_path = os.path.split(video_path)
    file_name, extention = file_path.split('.')
    output_folder = os.path.join(parent_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)

    extract_all_frames(video_path, output_folder)
