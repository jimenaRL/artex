import os
import time
import logging
import itertools
import subprocess
import concurrent.futures
from itertools import combinations
from glob import glob
from copy import deepcopy

import librosa
import numpy as np
from scipy.spatial import distance_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import ipdb
import warnings
warnings.filterwarnings('ignore')

NBFINALCOMBS = 6
MAXNBSTEPS = 10

SAMPLINGRATE = 44100
SEED = 1334
np.random.seed(SEED)


# from joblib import delayed, Parallel
# from joblib import dump, load


def compute_mfcc(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)
    return S, mfcc

def compute_mfcc_from_path(path):
    y, sr = librosa.load(path)
    return compute_mfcc(y, sr)

def show(path):
    S, mfccs = compute_mfcc_from_path(path)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                   x_axis='time', y_axis='mel', fmax=8000,
                                   ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.show()

def loadAndMiddleCrop(path, seconds=1):
    y, sr = librosa.load(path, sr=SAMPLINGRATE)
    if not sr == 44100:
        raise ValueError(f"sr: {sr}\npath: {path}")
    init = int( len(y) / 2 - 0.5 * seconds * sr )
    end = int( len(y) / 2 + 0.5 * seconds * sr )
    y = y[init:end]
    return y[:seconds*sr]

def getMeanAudio(comb):
    y0 = loadAndMiddleCrop(flutes["flute 0"][int(comb[0])])
    y1 = loadAndMiddleCrop(flutes["flute 1"][int(comb[1])])
    y2 = loadAndMiddleCrop(flutes["flute 2"][int(comb[2])])
    y3 = loadAndMiddleCrop(flutes["flute 3"][int(comb[3])])
    y4 = loadAndMiddleCrop(flutes["flute 4"][int(comb[4])])
    y5 = loadAndMiddleCrop(flutes["flute 5"][int(comb[5])])
    y = np.mean([y0, y1, y2, y3, y4, y5], axis=0)
    return y

def getMeanDescriptors(comb):
    y = getMeanAudio(comb)
    S, mfcc = compute_mfcc(y, sr=44100)
    return S, mfcc

def testDistenceMatrices(samples):
    # launch multiple threads computing distances and collect results
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_mfcc, s) for s in samples]
        results = [f.result() for f in futures]
    print(f"Calculing mffc fetures took {time.time() - start} seconds.")

    mel_spectrograms = [r[0].mean(axis=1) for r in results]
    mfccs = [r[1].mean(axis=1) for r in results]

    distances_mel = distance_matrix(mel_spectrograms, mel_spectrograms, p=2)
    distances_mfcc = distance_matrix(mfccs, mfccs, p=2)

    fig, ax = plt.subplots(figsize=(10,10))
    annot = True if nb_samples <= 10 else False
    heatmap = sns.heatmap(distances_mfcc, annot=annot, cmap='YlGnBu', linewidths=0.001, ax=ax)
    heatmap.set_title('mfcc')
    heatmap.get_figure().savefig(os.path.join(output_folder, f"distances_mfcc.png"))

    fig, ax = plt.subplots(figsize=(10,10))
    annot = True if nb_samples <= 10 else False
    heatmap = sns.heatmap(distances_mel, annot=annot, cmap='YlGnBu', linewidths=0.001, ax=ax)
    heatmap.set_title('mel')
    heatmap.get_figure().savefig(os.path.join(output_folder, f"distances_mel.png"))

def computeFeaturesFromCombinations(combinations, verbose=False):
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(getMeanDescriptors, comb) for comb in combinations]
        mean_decriptors = [f.result() for f in futures]
    # reorder
    mel_list = [m[0] for m in mean_decriptors]
    mfcc_list = [m[1] for m in mean_decriptors]
    if verbose:
        elapsed = time.time() - start
        estimation = 1000000 * elapsed / len(combinations)
        print(f"Calculing {len(combinations)} audio combinations took {elapsed} seconds.")
        print(f"Calculing 1000000 should take took {estimation / 3600} hours.")
    return mel_list, mfcc_list

if __name__ == "__main__":

    folder_name = 'renamed'
    audio_folder = f"/home/jimena/work/dev/ARTEX/{folder_name}"
    output_folder = "audio/features"
    os.makedirs(output_folder, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
      filename=os.path.join(output_folder, 'log.log'),
      filemode='w',
      level=logging.INFO)

    print(f"Extracting audio features from samples in {audio_folder}")

    samples = glob(os.path.join(audio_folder, '*.wav'))
    nb_samples = len(samples)
    print(f"Found {nb_samples} samples at {audio_folder}")
    # testDistenceMatrices(samples)

    flutes = {f"flute {k}": [] for k in range(6)}
    for sample in samples:
        label = f"flute {sample[-6]}"
        if label in flutes:
            flutes[label].append(sample)

    for f in flutes:
        print(f"Found {len(flutes[f])} audio samples for {f}")
        flutes[f] = flutes[f]

    combinations = [_ for _ in itertools.product(*['0123456789',]*6)]

    ###############
    # for dev
    # combinations = combinations[:1000]
    ###############

    nb_combs = len(combinations)
    print(f"There are {nb_combs} combinations possible")

    # test_mel, test_mfcc = computeFeaturesFromCombinations(combinations[:10])


    def getAuxDistanceMatrix(indexes):
        combs = [combinations[idx] for idx in indexes]
        mfcc = computeFeaturesFromCombinations(combs, verbose=False)[1]
        mfcc = np.array(mfcc).mean(axis=2)
        distances = distance_matrix(x=mfcc, y=mfcc)
        aux_distance = distances + 1e9 * np.eye(NBFINALCOMBS)
        return aux_distance

    print(f"------------------------- STEP -1")
    indexes = np.random.choice(nb_combs, size=NBFINALCOMBS, replace=False)
    aux_distance = getAuxDistanceMatrix(indexes)
    min_combs_dict = aux_distance.min()
    print(f"Current indexes are {indexes}")
    print(f"Current samples are {[combinations[idx] for idx in indexes]}")
    print(f"Current min distance is {min_combs_dict}")

    eject_a, eject_b = np.unravel_index(aux_distance.argmin(), aux_distance.shape)
    if aux_distance[eject_a].min() < aux_distance[eject_b].min():
        eject = int(eject_a)
    else:
        eject = int(eject_b)
    print(f"Ejecting candaidate is located at index {eject}")

    new_cadidate = np.random.choice(nb_combs, size=1, replace=False)
    print(f"New candidate is {new_cadidate}")

    new_indexes = deepcopy(indexes)
    new_indexes[eject] = new_cadidate
    new_aux_distance = getAuxDistanceMatrix(new_indexes)
    new_min_combs_dict = new_aux_distance.min()
    print(f"New indexes are {new_indexes}")
    print(f"New normalized min distance is {new_min_combs_dict}")

    if new_min_combs_dict > min_combs_dict:
        indexes = new_indexes
        print(f"Updating indexes to new ones")

    for _ in range(MAXNBSTEPS):
        print(f"------------------------- STEP {_}")
        aux_distance = getAuxDistanceMatrix(indexes)
        min_combs_dict = aux_distance.min()
        # print(f"Current indexes are {indexes}")
        # print(f"Current min distance is {min_combs_dict}")

        eject_a, eject_b = np.unravel_index(aux_distance.argmin(), aux_distance.shape)
        if aux_distance[eject_a].min() < aux_distance[eject_b].min():
            eject = int(eject_a)
        else:
            eject = int(eject_b)
        # print(f"Ejecting candaidate is located at index {eject}")

        new_cadidate = np.random.choice(nb_combs, size=1, replace=False)
        # print(f"New candidate is {new_cadidate}")

        new_indexes = deepcopy(indexes)
        new_indexes[eject] = new_cadidate
        new_aux_distance = getAuxDistanceMatrix(new_indexes)
        new_min_combs_dict = new_aux_distance.min()
        # print(f"New indexes are {new_indexes}")
        # print(f"New min distance is {new_min_combs_dict}")

        if new_min_combs_dict > min_combs_dict:
            indexes = new_indexes
            print(f"Updating indexes to new ones. New min distance is {new_min_combs_dict}")


    final_combinations = [combinations[i] for i in indexes]
    print(f"Final indexes are: {indexes}")
    print(f"Final combinations are: {final_combinations}")
    final_mfcc = computeFeaturesFromCombinations(final_combinations, verbose=False)[1]

    fig, axes = plt.subplots(nrows=NBFINALCOMBS, sharex=True)
    for ax, mfccs, comb in zip(axes, final_mfcc, final_combinations):
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
        fig.colorbar(img, ax=[ax])
        ax.set(title=f'Final MFCC tuples {''.join(comb)}')
    plt.show()
    fig.savefig(f"final_mfcc_NBFINALCOMBS_{NBFINALCOMBS}_MAXNBSTEPS_{MAXNBSTEPS}_SEED_{SEED}.png")


    with open(f"final_mfcc_NBFINALCOMBS_{NBFINALCOMBS}_MAXNBSTEPS_{MAXNBSTEPS}_SEED_{SEED}.txt", 'w') as f:
        f.writelines('\n'.join([','.join(c) for c in final_combinations])+'\n')