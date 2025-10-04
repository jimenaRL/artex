import os
import json
import time
import itertools
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from scipy.spatial import distance_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULTNBFINALIMAGES = 10
DEFAULTMAXNBSTEPS = 500
DEFAULTSEED = 1334

# DEFAULTRESULTSFOLDER = "/home/jimena/work/dev/artex/results/ForLearning_2025-10-03_19-47-53"
# DEFAULTIMAGEFOLDER = "/home/jimena/work/dev/artex/ForLearning_2025-10-03_19-47-53/images"

DEFAULTRESULTSFOLDER = "/home/jimena/work/dev/ARTEX/results/ForLearning_2025-10-03_19-47-53"
DEFAULTIMAGEFOLDER = "/home/jimena/work/dev/ARTEX/ForLearning_2025-10-03_19-47-53/images"


# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/dino/issues/72
DEFAULTREPO = 'facebookresearch/dino:main'
DEFAULTMODEL = 'dino_resnet50'

import ipdb
import warnings
warnings.filterwarnings('ignore')

def compute_features(model, path):
    preprocess = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_features(model, image):
      return model(preprocess(image).unsqueeze(0))

    return get_features(resnet50, Image.open(path)).detach().numpy()

def allImagesDistanceMatrix(embeddings, name):
    distances = distance_matrix(x=embeddings, y=embeddings)
    print(f"distance:\n{distances}")

    npfile = f"{name}.npy"
    with open(npfile, 'wb') as f:
        np.save(f, distances)
    print(f"Distance matrix array saved at: {npfile}")

    csvfile = f"{name}.csv"
    pd.DataFrame(distances).to_csv(csvfile)
    print(f"Distance matrix csv saved at: {csvfile}")

    fig, ax = plt.subplots(figsize=(10,10))
    annot = True if nb_images <= 10 else False
    heatmap = sns.heatmap(distances, annot=annot, cmap='YlGnBu', linewidths=0.001, ax=ax)

    imgfile = f"{name}.png"
    heatmap.get_figure().savefig(imgfile)
    print(f"Distance matrix image saved at: {imgfile}")

    return distances


def pairsFind(distmatrix, allembeddings, finalnbimages):

    nb_images = allembeddings.shape[0]
    indexes = np.random.choice(nb_images, size=finalnbimages, replace=False).tolist()

    embeddings = allembeddings[indexes]
    aux_distance = distance_matrix(x=embeddings, y=embeddings) + 1e9 * np.eye(finalnbimages)
    min_distance = aux_distance.min()
    print(f"Current indexes are {indexes}")
    print(f"Current min distance is {min_distance}")

    eject_a, eject_b = np.unravel_index(aux_distance.argmin(), aux_distance.shape)
    if aux_distance[eject_a].min() < aux_distance[eject_b].min():
        eject = int(eject_a)
    else:
        eject = int(eject_b)
    print(f"Ejecting candidate is located at index {eject}")

    new_cadidate = np.random.choice(nb_images, size=1, replace=False).tolist()[0]
    print(f"New candidate is {new_cadidate}")

    new_indexes = deepcopy(indexes)
    new_indexes[eject] = new_cadidate
    new_embeddings = allembeddings[new_indexes]
    new_aux_distance = distance_matrix(x=new_embeddings, y=new_embeddings) + 1e9 * np.eye(finalnbimages)
    new_min_distance = new_aux_distance.min()
    print(f"New indexes are {new_indexes}")
    print(f"New  min distance is {new_min_distance}")

    if new_min_distance > min_distance:
        indexes = new_indexes
        print(f"Updating indexes to new ones")

    for _ in tqdm(range(nbsteps)):
        embeddings = allembeddings[indexes]
        aux_distance = distance_matrix(x=embeddings, y=embeddings) + 1e9 * np.eye(finalnbimages)
        min_distance = aux_distance.min()

        eject_a, eject_b = np.unravel_index(aux_distance.argmin(), aux_distance.shape)
        if aux_distance[eject_a].min() < aux_distance[eject_b].min():
            eject = int(eject_a)
        else:
            eject = int(eject_b)
        # print(f"Ejecting candidate is located at index {eject}")

        new_cadidate = np.random.choice(nb_images, size=1, replace=False).tolist()[0]
        # print(f"New candidate is {new_cadidate}")

        new_indexes = deepcopy(indexes)
        new_indexes[eject] = new_cadidate
        new_embeddings = allembeddings[new_indexes]
        new_aux_distance = distance_matrix(x=new_embeddings, y=new_embeddings) + 1e9 * np.eye(finalnbimages)
        new_min_distance = new_aux_distance.min()
        # print(f"New indexes are {new_indexes}")
        # print(f"New  min distance is {new_min_distance}")

        if new_min_distance > min_distance:
            indexes = new_indexes
            min_distance = new_min_distance
            print(f"New min distance found at step {_}")
            print(f"Updating indexes to new ones. New min distance is {min_distance}")

    print(f"Final indexes are: {indexes}")
    print(f"Final min distance is: {min_distance}")

    return indexes

if __name__ == "__main__":

    ap = ArgumentParser()
    ap.add_argument('--image_folder', type=str, default=DEFAULTIMAGEFOLDER)
    ap.add_argument('--repository', type=str, default=DEFAULTREPO)
    ap.add_argument('--model', type=str, default=DEFAULTMODEL)
    ap.add_argument('--finalnbimages', type=int, default=DEFAULTNBFINALIMAGES)
    ap.add_argument('--nbsteps', type=int, default=DEFAULTMAXNBSTEPS)
    ap.add_argument('--resultsfolder', type=str, default=DEFAULTRESULTSFOLDER)
    ap.add_argument('--seed', type=int, default=DEFAULTSEED)

    args = ap.parse_args()
    image_folder = args.image_folder
    repository = args.repository
    model = args.model
    finalnbimages = args.finalnbimages
    nbsteps = args.nbsteps
    resultsfolder = args.resultsfolder
    seed = args.seed


    parameters = vars(args)
    dumped_parameters = json.dumps(parameters, sort_keys=True, indent=4)
    print("\n\n---------------------------------------------------------")
    print(f"PARAMETERS:\n{dumped_parameters[2:-2]}")
    print("---------------------------------------------------------")

    np.random.seed(seed)
    print(f"Computing embeddings from images in {image_folder} using {model}.")

    allimages = glob(os.path.join(image_folder, '*.png'))
    allimages.sort()

    embfile = os.path.join(resultsfolder, f"frame_embeddings.npy")
    filesname = os.path.join(resultsfolder, f"emb_distance_matrix")
    disfile = f"{filesname}.npy"
    if os.path.exists(embfile):
        print(f"Loading embeddings from {embfile}")
        with open(embfile, 'rb') as f:
            allembeddings = np.load(f)
        print(f"Loading distance matrix from {disfile}")
        with open(disfile, 'rb') as f:
            alldistances = np.load(f)
    else:

        import torch
        from torchvision import transforms as pth_transforms

        resnet50 = torch.hub.load(repo_or_dir=repository, model=model)
        start = time.time()
        allembeddings = []

        for image in tqdm(allimages):
            allembeddings.append(compute_features(resnet50, image))
        nb_images = np.array(allembeddings).squeeze()
        print(f"Calculing {len(nb_images)} embeddings took {time.time() - start} seconds.")
        print(f"Saving embeddings to {embfile}")

        with open(embfile, 'wb') as f:
            np.save(f, allembeddings)
        imgfile = os.path.join(resultsfolder, f"{video}_emb_distance_matrix.png")
        alldistances = allImagesDistanceMatrix(allembeddings, filesname)

        plt.hist(alldistances.flatten(), bins='auto')
        plt.title("Distances histogram")
        plt.show()

    print(f"Embeddings shape are: {allembeddings.shape}")

    print("\n\n---------------------------------------------------------")
    print(f"PAIRS FIND ALGO")
    indexes = pairsFind(alldistances, allembeddings, finalnbimages)
    print("---------------------------------------------------------")

    for i in indexes:
        print(allimages[i])

    name = f"selected_images_NBIMAGES_{finalnbimages}_MAXNBSTEPS_{nbsteps}_SEED_{seed}"
    selected_images_folder = os.path.join(resultsfolder, name)
    os.makedirs(selected_images_folder, exist_ok=True)
    for i in indexes:
        os.system(f"cp {allimages[i]} {selected_images_folder}/")

    os.system(f"open {selected_images_folder}/")
