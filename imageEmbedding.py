# https://dinov2.metademolab.com/


from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("ForLearning_2025/images/000132.png")

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

inputs = processor(images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state


print(last_hidden_states)


# https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m

from transformers import pipeline
from transformers.image_utils import load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-vit7b16-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
features = feature_extractor(image)

print(features)


# https://github.com/facebookresearch/dino/issues/72

from PIL import Image
import requests

def get_image(url):
  return Image.open(requests.get(url, stream=True).raw)
from torchvision import transforms as pth_transforms

preprocess = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_features(model, image):
  return model(preprocess(image).unsqueeze(0))
import torch

resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
features = get_features(resnet50, get_image(url))


################

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
        '--video', type=str, default="ForLearning_2025.mp4")
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
    embeddings = 
    logger.info(f"Calculing earth movers distances took {time.time() - start} seconds.")

    # store results in a matrix
    emb_distances = np.zeros((nb_images, nb_images))
    for n, p in enumerate(pairs):
        i = int(os.path.split(p[0])[1].split(".png")[0]) - 1
        j = int(os.path.split(p[1])[1].split(".png")[0]) - 1
        emb_distances[i, j] = embeddings[n]

    emb_distances = emb_distances + emb_distances.T
    emb_distances[np.diag_indices(nb_images)] = 1

    print(f"earth movers distance:\n{emb_distances}")

    np.savetxt(
      X=emd_distances,
      fname=os.path.join(output_folder, f"{file_name}_emb.csv"),
      delimiter=",")

    fig, ax = plt.subplots(figsize=(10,10))
    annot = True if nb_images <= 10 else False
    heatmap = sns.heatmap(emd_distances, annot=annot, cmap='YlGnBu', linewidths=0.001, ax=ax)
    heatmap.get_figure().savefig(os.path.join(output_folder, f"{file_name}_emb.png"))