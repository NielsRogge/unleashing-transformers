
from models import Generator
from hparams import get_sampler_hparams
from train_sampler import get_sampler
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_samples
from utils.log_utils import load_model


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

    # step 3: sample
    samples = get_samples(H, generator, sampler)

    return samples


if __name__ == "__main__":
    H = get_sampler_hparams()
    main(H)