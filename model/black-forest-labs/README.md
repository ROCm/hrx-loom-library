# Black Forest Labs models

Models trained by [Black Forest Labs](https://huggingface.co/black-forest-labs).
Quantized derivatives remain under the original base model's directory and are
named card variants.

[FLUX.2 [klein] 4B](FLUX.2-klein-4B/OVERVIEW.md) supplies a four-step
text-to-image composition: a Qwen3 text encoder, a distilled diffusion
transformer with joint and single blocks, and a tiled VAE decoder. The package
has source programs, numerical tests, and a served image check.
