
from models import Generator
from hparams import get_sampler_hparams
from train_sampler import get_sampler
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_samples
from utils.log_utils import load_model

import torch

from torchvision import utils


def main(H):
    # step 1: load the quantizer + generator of the VQ-VAE
    quantizer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ["quantize", "generator"],
        remove_component_from_key=True
    )
    embedding_weight = quantizer_and_generator_state_dict.pop("embedding.weight")
    embedding_weight = embedding_weight.cuda()
    
    generator = Generator(H)
    generator.load_state_dict(quantizer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    
    # step 2: load the sampler
    sampler = get_sampler(H, embedding_weight).cuda()
    sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir)
    sampler = sampler.cuda()

    # test dummy input on Transformer denoiser
    dummy_input = torch.tensor([[1024,1024,1024]]).cuda()
    transformer_out = sampler._denoise_fn(dummy_input, t=2)
    print("Shape of transformer out:", transformer_out.shape)
    print("First values of transformer out:", transformer_out[0, :3, :3])

    # step 3: sample
    print("Sampling images...")
    images = get_samples(H, generator, sampler)

    utils.save_image(images, "results.png", nrow = H.batch_size)

    return images


if __name__ == "__main__":
    H = get_sampler_hparams()
    main(H)