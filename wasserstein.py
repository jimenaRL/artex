import torch
from transformers import AutoImageProcessor, AutoModel


model_ckpt = "nateraw/vit-base-beans"
processor = AutoImageProcessor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

def extract_embeddings(model: torch.nn.Module):
    """Utility to compute embeddings."""
    device = model.device

    def pp(batch):
        images = batch["image"]
        # `transformation_chain` is a compostion of preprocessing
        # transformations we apply to the input images to prepare them
        # for the model. For more details, check out the accompanying Colab Notebook.
        image_batch_transformed = torch.stack(
            [transformation_chain(image) for image in images]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()
        return {"embeddings": embeddings}

    return pp



if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_fn = extract_embeddings(model.to(device))
    candidate_subset_emb = candidate_subset.map(extract_fn, batched=True, batch_size=batch_size)