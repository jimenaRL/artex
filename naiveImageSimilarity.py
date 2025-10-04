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

##
# Globals
##

warnings.filterwarnings('ignore')

# specify resized image sizes
height = 2**10
width = 2**10

##
# Functions
##

def get_img(path, norm_size=False, norm_exposure=False):
  '''
  Prepare an image for image processing tasks
  '''
  # flatten returns a 2d grayscale array
  img = cv2.imread(path)
  # make image grayscale
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # resizing returns float vals 0:255; convert to ints for downstream tasks
  # if norm_size:
  # if norm_exposure:
  return img

def get_histogram(img):
  '''
  Get the histogram of an image.
  The histogram is a vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  bgr_planes = cv2.split(img)
  histSize = 256
  accumulate = False
  histg = cv2.calcHist(bgr_planes, [0], None, [histSize], (0, histSize), accumulate=False)
  return histg.flatten()

def earth_movers_distance(path_a, path_b):
  '''
  Measure the Earth Mover's distance between two images
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=False)
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)
  return wasserstein_distance(hist_a, hist_b)


def l0(path_a, path_b):
  '''
  Measure the pixel-level similarity between two images
  '''
  img_a = get_img(path_a, norm_exposure=True)
  img_b = get_img(path_b, norm_exposure=True)
  return np.sum(np.absolute(img_a - img_b)) / (height*width)

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
    ap.add_argument('--limit', type=int, default=3)

    args = ap.parse_args()
    video_path = args.video
    limit = args.limit

    parent_folder, file_path = os.path.split(video_path)
    file_name, extention = file_path.split('.')
    output_folder = os.path.join(parent_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)
    output_folder_images = os.path.join(parent_folder, file_name, "images")
    os.makedirs(output_folder_images, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
      filename=os.path.join(output_folder, 'log.log'),
      filemode='w',
      level=logging.INFO)

    logger.info(f"Extranting images from video {video_path} to folder {output_folder}")
    extract_all_frames(video_path, output_folder_images)

    images = glob(os.path.join(output_folder_images, '*'))[:limit]
    nb_images = len(images)
    logger.info(f"Found {nb_images} images at {output_folder}")
    logger.info(f"Must compute {int(nb_images * (nb_images - 1) / 2)} distances.")

    pairs = [p for p in itertools.combinations(images, 2)]

    # launch multiple threads computing distances and collect results
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        l0_futures = [executor.submit(l0, p[0], p[1]) for p in pairs]
        l0_results = [f.result() for f in l0_futures]
    logger.info(f"Calculing l0 distances took {time.time() - start} seconds.")

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        emd_futures = [executor.submit(earth_movers_distance, p[0], p[1]) for p in pairs]
        emd_results = [f.result() for f in emd_futures]
    logger.info(f"Calculing earth movers distances took {time.time() - start} seconds.")

    # store results in a matrix
    l0_distances = np.zeros((nb_images, nb_images))
    emd_distances = np.zeros((nb_images, nb_images))
    for n, p in enumerate(pairs):
        i = int(os.path.split(p[0])[1].split(".png")[0]) - 1
        j = int(os.path.split(p[1])[1].split(".png")[0]) - 1
        l0_distances[i, j] = l0_results[n]
        emd_distances[i, j] = emd_results[n]

    l0_distances = l0_distances + l0_distances.T
    l0_distances[np.diag_indices(nb_images)] = 1

    emd_distances = emd_distances + emd_distances.T
    emd_distances[np.diag_indices(nb_images)] = 1

    # print(f"l0:\n{l0_distances}")
    # print(f"earth movers distance:\n{emd_distances}")

    np.savetxt(
      X=l0_distances,
      fname=os.path.join(output_folder, f"{file_name}_l0.csv"),
      delimiter=",")
    np.savetxt(
      X=emd_distances,
      fname=os.path.join(output_folder, f"{file_name}_emd.csv"),
      delimiter=",")

    fig, ax = plt.subplots(figsize=(10,10))
    annot = True if nb_images <= 10 else False
    heatmap = sns.heatmap(emd_distances, annot=annot, cmap='YlGnBu', linewidths=0.001, ax=ax)
    heatmap.get_figure().savefig(os.path.join(output_folder, f"{file_name}_emd.png"))

    fig, ax = plt.subplots(figsize=(10,10))
    annot = True if nb_images <= 10 else False
    heatmap = sns.heatmap(l0_distances, annot=annot, cmap='crest', linewidths=0.001, ax=ax)
    heatmap.get_figure().savefig(os.path.join(output_folder, f"{file_name}_l0.png"))

