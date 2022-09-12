
from models import VQAutoEncoder
from hparams import get_sampler_hparams
from train_sampler import get_sampler
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_samples
from utils.log_utils import load_model

import torch

from PIL import Image
import requests

from torchvision.transforms import Compose, Resize, ToTensor, Normalize


image_transformations = Compose([Resize((256,256)),
                                 ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def main(H):
    # step 1: load the VQ-VAE
    model = VQAutoEncoder(H)
    ae_load_path = f"{H.ae_load_dir}/saved_models/vqgan_ema_{H.ae_load_step}.th"
    state_dict = torch.load(ae_load_path, map_location="cpu")
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