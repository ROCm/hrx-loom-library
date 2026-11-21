## Upscale an image

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) x4plus upscales an image
four times on each axis with an RRDBNet generator of 23 residual-in-residual
dense blocks. This Loom package runs that network on AMD GPUs using the native
libamdf driver. It reads the original F32 checkpoint directly; weight packing
and specialization happen on the device during model preparation and
execution.

Start with an installed `hrx-loom` SDK and its optional `hf` extra. The
[SDK installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
describes supported systems and installation. These commands need no source
build or Bazel installation:

```sh
model_url=https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/xinntao/Real-ESRGAN-x4plus
loom-fetch "$model_url/model.json" --list
loom-fetch "$model_url/model.json" --preset=upscale --output=./real-esrgan-x4plus
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 --model=./real-esrgan-x4plus/model.json
```

Fetch reads the adjacent acquisition lock, verifies the selected files, and
writes a local card with the selected preset. It shares cached checkpoint
storage by default. The server uses local files and performs no downloads.
Its listener becomes available before asynchronous model preparation finishes;
`GET /v1/status` reports preparation progress and errors, and `GET /v1/models`
lists ready models.

The image service is text to image and has no image-input field, so this card
takes its input in `prompt`: base64 of the input's raw 8-bit RGB pixels,
row-major with no header. `size` is the output, four times the input width and
height. Standard (`+/`) and URL-safe (`-_`) base64 are both accepted. The
response is an RGBA PNG whose alpha is 255 everywhere. In another terminal,
this Python example uses [Pillow](https://pypi.org/project/pillow/) to send
`input.png` and save the result:

```python
import base64, json, urllib.request
from PIL import Image

image = Image.open("input.png").convert("RGB")
width, height = image.size
request = {
    "model": "xinntao/Real-ESRGAN-x4plus",
    "prompt": base64.b64encode(image.tobytes()).decode(),
    "size": f"{4 * width}x{4 * height}",
    "response_format": "b64_json",
}
reply = urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8080/v1/images/generations", json.dumps(request).encode(),
    {"Content-Type": "application/json"}))
with open("upscaled.png", "wb") as output:
    output.write(base64.b64decode(json.load(reply)["data"][0]["b64_json"]))
```

Real-ESRGAN has no sampling: `seed` is accepted and ignored, and the same input
always produces the same pixels. The
[image-generation protocol](https://rocm.github.io/hrx-system/loom/serving/http-profiles/#image-generation)
describes the remaining request fields and error responses. Ctrl-C stops
admission and joins accepted work before the server exits.

## Choose a preset

| Preset | Default input | Default output |
| --- | --- | --- |
| `upscale` | 256×256 | 1024×1024 |
| `lifecycle` | 64×64 | 256×256 |

Both presets read the same checkpoint. The preset geometry is prepared when
the server starts; another supported geometry is prepared on its first
request and then retained.

## Exact upstream files

The publisher distributes the checkpoint only as the pickled
`RealESRGAN_x4plus.pth` on the
[v0.1.0 release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0),
which `loom-fetch` does not read. The `f32` variant instead reads the
safetensors copy from
[Comfy-Org/Real-ESRGAN](https://huggingface.co/Comfy-Org/Real-ESRGAN/tree/5fd49b7b278836f48af63ecd314d0f98ab336105):

- `RealESRGAN_x4plus.safetensors`

That file holds the original 702 F32 tensors under their original names; each
tensor matches the release `.pth` exactly. It loads into `parameters.esrgan`.
The convolution weights are rounded to BF16 on the device, and activations,
biases, accumulation, and residual arithmetic stay F32.

The image service requires a text tokenizer, but the model reads no text. The
card therefore ships its own `tokenizer.json`, a vocabulary of the base64
alphabet, its URL-safe characters, and `=`. It lets the server accept the
prompt; the programs read the prompt bytes directly and never use token ids.

Real-ESRGAN and its weights are published under the
[BSD 3-Clause License](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE).
The [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN) and paper
([arXiv:2107.10833](https://arxiv.org/abs/2107.10833)) describe training,
intended uses, and limitations.

## Geometry and program choices

The output width must be a multiple of 256 from 256 to 4096, and the output
height a multiple of 16 from 16 to 4096, so the input is a multiple of 64
pixels wide and 4 pixels high, at most 1024 on each axis. Non-square images
are supported. The prompt must hold exactly one input pixel per four base64
characters; a prompt whose length disagrees with `size`, or that contains any
other character, is rejected before device work. Crop or pad other input
sizes first. Unsupported geometry fails instead of silently resizing the
output.

The whole image runs in one pass, so there are no tile seams. Deployment
options, adapters, and more than one image per request are rejected. An alpha
channel in the input is not upscaled; send RGB.

The source publication contains Loom text providers and an acquisition lock.
Fetch selects it automatically. Authors can also build bytecode publications;
both retain multiple linkable providers and use the same runtime specialization
contract. `--programs` chooses among published sets; `--preset` chooses
execution settings. Consumers of the source publication do not need to build
bytecode.
