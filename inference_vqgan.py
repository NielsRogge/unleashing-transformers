
from models import VQAutoEncoder
from hparams import get_sampler_hparams
from train_sampler import get_sampler
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_samples
from utils.log_utils import load_model

import torch

from PIL import Image
import requests

from torchvision.transforms import Compose, Resize, ToTensor, Normalize


image_transformations = Compose([Resize, ToTensor, Normalize])


def main(H):
    # step 1: load the VQ-VAE
    model = VQAutoEncoder(H)
    state_dict = torch.load(H.ae_load_dir, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    
    # step 2: run forward pass on cats image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    pixel_values = image_transformations(image).unsqueeze(0)
    out = model.encoder(pixel_values)
    print("Shape of out:", out.shape)

    return out


if __name__ == "__main__":
    H = get_sampler_hparams()
    main(H)