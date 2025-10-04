# https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c

import ipdb

import os
import time
import logging
import warnings
import itertools
import subprocess
import concurrent.futures
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

import cv2
import numpy as np
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt


def extract_all_frames(video_path, output_folder):
    """
    Extracts high-quality JPEG image from a video using FFmpeg.
    """

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
        '--video', type=str, default="../ForLearning_2025.mp4")

    args = ap.parse_args()
    video_path = args.video

    parent_folder, file_path = os.path.split(video_path)
    file_name, extention = file_path.split('.')
    output_folder = os.path.join(parent_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)
    output_folder_images = os.path.join(parent_folder, file_name, "images")
    os.makedirs(output_folder_images, exist_ok=True)

    print(f"Extranting images from video {video_path} to folder {output_folder}")
    extract_all_frames(video_path, output_folder_images)

    images = glob(os.path.join(output_folder_images, '*'))
    nb_images = len(images)
    print(f"Wrote {nb_images} images at {output_folder}")
