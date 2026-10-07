<div align="center">
<h1>🎮 WorldPlay2: Extending Real-Time Interactive World Models in Control and Horizon</h1>
</div>

<div align="center">
  <a href=https://worldplay2.github.io/ target="_blank"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=homepage height=22px></a>
  <a href=https://arxiv.org/abs/2609.35560 target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://www.youtube.com/watch?v=_pFzHHslhgc target="_blank"><img src=https://img.shields.io/badge/YouTube%20Video-FF0000.svg?logo=youtube height=22px></a>
  <a href="https://reactor.inc/worldplay2" target="_blank"><img src="reactor/assets/try-on-reactor.svg" alt="Try it on Reactor" width="128" height="22"></a>
</div>


## 📰 News

- **[2026.10.7]** WorldPlay2 is now on Reactor! [Play in your browser](https://reactor.inc/worldplay2), or [deploy it locally](reactor/README.md) with the Reactor integration and demo frontend in this repository. Thanks to [Rising0321](https://github.com/Rising0321), [Orion-Zheng](https://github.com/Orion-Zheng), and [notrealzapa](https://github.com/notrealzapa) for porting WorldPlay2 to Reactor and optimizing the inference infrastructure.
- **[2026.10.6]** We have released the WorldPlay2 model weights and inference code!

## 🎥 Video

https://github.com/user-attachments/assets/3264efa7-ef6a-4c7b-a043-d819264a93bd

## 🛠️ Installation

Requires Python 3.10+ and a CUDA-capable NVIDIA GPU.

Create and activate a Conda environment, then install the dependencies:

```bash
conda create -n worldplay2 python=3.10 -y
conda activate worldplay2
python -m pip install -r requirements.txt
```

Install `flash_attn`:

```bash
pip install flash-attn --no-build-isolation
```

#### Optional SageAttention

Install SageAttention in your CUDA environment:

```bash
pip install sageattention==2.2.0 --no-build-isolation
```

All three launchers support `ATTENTION_BACKEND` (default: `flash`). For direct
CLI usage, pass `--attention_backend sage`.

#### Optional FP8

Install the additional dependencies on supported hardware:

```bash
pip install transformer_engine[pytorch]
```

After setting the model and input paths in `run_few_step.sh` or exporting them
as environment variables, enable FP8:

```bash
ENABLE_FP8_FFN=true ENABLE_FP8_QKV=true USE_DIT_FSDP=false bash run_few_step.sh
```

## 🧱 Model Download

### Base Model

Download [Wan2.2 I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
and set `CKPT_DIR` to its local directory.

### WorldPlay2 Models

Download the WorldPlay2 checkpoints for your inference mode. Set
`LOW_NOISE_CKPT` and `HIGH_NOISE_CKPT` to the corresponding checkpoint files.

<!-- TODO: Replace each # below with the corresponding Hugging Face model URL. -->

| Model               | Model Type                          | Default steps | Download Links |
|---------------------|-------------------------------------|---------------| --- |
| **WorldPlay2-Fast** | Autoregressive, 4-step (`few_step`) | 4             | [🤗 Hugging Face](https://huggingface.co/aejion/WorldPlay2-Fast) |
| **WorldPlay2-BI**   | Bidirectional (`bi`)                | 40            | [🤗 Hugging Face](https://huggingface.co/aejion/WorldPlay2-BI) |
| **WorldPlay2-AR**   | Autoregressive (`ar`)               | 40            | [🤗 Hugging Face](https://huggingface.co/aejion/WorldPlay2-AR) |
| **WorldPlay2-TAE**  | Causal VAE                          | ---           | (coming soon) |

Download the base model and WorldPlay2 few-step model using the Hugging Face CLI.

```bash
pip install "huggingface_hub[cli]>=0.34,<1.0"

# Base Wan2.2 model
hf download Wan-AI/Wan2.2-I2V-A14B --local-dir ./checkpoints/Wan2.2-I2V-A14B

# WorldPlay2 models
hf download aejion/WorldPlay2-Fast --local-dir ./checkpoints/WorldPlay2-Fast
hf download aejion/WorldPlay2-BI --local-dir ./checkpoints/WorldPlay2-BI
hf download aejion/WorldPlay2-AR --local-dir ./checkpoints/WorldPlay2-AR
```

Set `CKPT_DIR` to `./checkpoints/Wan2.2-I2V-A14B`, and set `LOW_NOISE_CKPT`
and `HIGH_NOISE_CKPT` to the checkpoint files in `./checkpoints/WorldPlay2-Fast`.

## 🎮 Quick Start

Set the paths to the WorldPlay2 few-step checkpoints and run from the repository root:

```bash
# CKPT_DIR: path to the original Wan2.2 I2V-A14B base model directory.
# LOW_NOISE_CKPT: path to the WorldPlay2 few-step low-noise checkpoint.
# HIGH_NOISE_CKPT: path to the WorldPlay2 few-step high-noise checkpoint.
CKPT_DIR=/path/to/model \
LOW_NOISE_CKPT=/path/to/model/low_noise_model.safetensors \
HIGH_NOISE_CKPT=/path/to/model/high_noise_model.safetensors \
INPUT_JSON=/path/to/input.json \
OUTPUT_PATH=./outputs/few_step \
bash run_few_step.sh
```

Use `run_ar.sh` or `run_bi.sh` for the other modes, with their corresponding
checkpoints. Launchers default to 8 GPUs with sequence parallelism and FSDP;
set `NPROC_PER_NODE` to match your GPU count. For one GPU, also set
`USE_DIT_FSDP=false USE_T5_FSDP=false`.

All input items are processed, and videos are saved as `<output_name>.mp4` in
`OUTPUT_PATH`.

## ⚡ Quick Start with Reactor Runtime

The [`reactor/`](reactor/README.md) integration serves WorldPlay2 as an interactive model with a browser frontend. The default configuration uses 4 NVIDIA B200 GPUs, Docker and the NVIDIA Container Toolkit. Follow the [Reactor setup guide](reactor/README.md#1-install-the-reactor-cli) to install the CLI and prepare the weights under `reactor/weights/base` and `reactor/weights/fast`.

From the repository root, enter `reactor/`, then build and start the model on four available GPUs:

```bash
cd reactor
reactor build
reactor run --gpus '"device=0,1,2,3"' --port 8080
```

Once `http://localhost:8080/health` reports `"state":"available"`, open another terminal at the repository root and start the frontend (Node.js 20+ and pnpm required):

```bash
cd reactor/demo
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000), select the **Local** endpoint (`http://localhost:8080`), and click **Connect**. Choose an example or add your own image and prompt to start playing, then use **WASD** to move and the **arrow keys** to look around. Release the keys or on-screen buttons to stop the corresponding action. For a remote GPU machine, see the [connection instructions](reactor/README.md#4-open-the-client), including WebRTC requirements for SSH access.

## 📖 Input Format

Provide a JSON array. Relative image paths are resolved from the JSON file's
directory.

```json
[
  {
    "image_path": "input.jpg",
    "perspective": "tps",
    "action": "w-32,s-32",
    "prompt_event": "prompt1-32,prompt2-32",
    "prompt1": "The scene in the video is: a quiet forest trail. The character is: a hiker wearing a blue jacket. The event is: dark clouds gather and rain begins to fall.",
    "prompt2": "The scene in the video is: a quiet forest trail. The character is: a hiker wearing a blue jacket. The event is: the rain stops and sunlight breaks through the clouds.",
    "output_name": "sample_000"
  }
]
```

- **Perspective:** `fps` or `tps`.
- **Actions:** comma-separated `action-duration` commands. Movement actions are
  `w`, `s`, `a`, `d`, `wa`, `wd`, `sa`, `sd`; rotation actions are `left`, `right`,
  `up`, `down`; special action is `space`. Combine tokens with `+`, for example `right+space-32`.
- **Durations:** measured in latent frames; the first latent is stationary.
  Action and prompt durations must have the same total. The total and prompt
  boundaries must align with `chunk_length` (4 in the few-step and AR launchers,
  32 in BI).

#### Prompt Format

Construct each prompt field (`prompt1`, `prompt2`, etc.) using the template for
your perspective, replacing the placeholders with your descriptions.

**FPS (`fps`):** omit the `The character is: ...` sentence.

```text
The scene in the video is: {scene description}. The event is: {event description}.
```

**TPS (`tps`):** include the character description, as in the JSON example above.

```text
The scene in the video is: {scene description}. The character is: {character description}. The event is: {event description}.
```

The scene description is required for both perspectives; the character
description is only required for TPS. The event is optional for both: omit
the entire `The event is: ...` sentence when it is not needed. For example,
an FPS prompt without an event is:

```text
The scene in the video is: a quiet forest trail.
```

The `prompt_event` scheduling field is still required, even when prompts omit
the event description.

## 🔑 Configuration

Paths and launcher settings can be overridden through environment variables:

| Variable | Purpose |
| --- | --- |
| `NPROC_PER_NODE` | Number of GPUs |
| `ATTENTION_BACKEND` | Visual self-attention: `flash` (default) or `sage` |
| `SAMPLE_STEPS` | Sampling steps; few-step inference requires 4 |
| `GUIDE_SCALE` | Guidance scale for BI/AR |
| `YAW_ROTATION_SPEED_DEG` | Left/right rotation per latent |
| `PITCH_ROTATION_SPEED_DEG` | Up/down rotation per latent |

## 🙏 Acknowledgements
We would like to thank [HY-World 1.5](https://github.com/Tencent-Hunyuan/HY-WorldPlay), [HY-World 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0), and [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) for their great work.
