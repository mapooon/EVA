# Third-party notices

## MAR

- Repository: <https://github.com/LTH14/mar>
- License: MIT
- Copyright: Tianhong Li
- Usage: EVA is based on the MAR repository structure and training utilities, and contains
  code derived from MAR.

The original MAR copyright and permission notice is retained in [LICENSE](LICENSE).

## Latent Diffusion KL-VAE implementation

- Repository: <https://github.com/CompVis/latent-diffusion>
- Usage: The ImageNet tokenizer implementation in `models/image_vae.py` is adopted from the
  KL-VAE implementation in Latent Diffusion.

## MAR-distributed KL-16 VAE checkpoint

- Repository: <https://github.com/LTH14/mar>
- Distribution: The pretrained KL-16 VAE checkpoint used by EVA was distributed by MAR through
  the download utility in the MAR repository.
- Usage: ImageNet tokenizer weights and the corresponding latent scale.
- EVA does not relicense this pretrained checkpoint; the applicable upstream terms remain in
  effect.

## Stable Audio Open

- Model: <https://huggingface.co/stabilityai/stable-audio-open-1.0>
- Usage: VGGSound tokenizer.
- The Stable Audio Open model license applies to the tokenizer and its use.

## Datasets and evaluation models

ImageNet, VGGSound, VGGish, and other evaluation assets are not relicensed by this repository.
Users must comply with the original dataset, model, and asset terms when downloading or using
them.
