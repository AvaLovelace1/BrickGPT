---
title: BrickGPT-Demo
emoji: 🧱
colorFrom: green
colorTo: yellow
license: mit
short_description: Generate stable toy brick structures from text prompts.
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
python_version: 3.10.13
models:
  - AvaLovelace/BrickGPT
pinned: true
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/672403d5f328a3e6638331ee/H66Srg95-N--44lg9-4dX.png
---

# BrickGPT Demo

This subdirectory contains the code for the official BrickGPT Gradio demo.

## Prerequisites

- **Llama-3.2-1B-Instruct:** BrickGPT is fine-tuned from meta-llama/Llama-3.2-1B-Instruct, a gated model. Request access
  to the model [here](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct). Then generate
  a [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) and set your access token as
  an environment variable: `export HF_TOKEN=<your_token>`.
  The model will be automatically downloaded upon running the code.
- **Gurobi:** Running stability analysis requires a [Gurobi licence](https://www.gurobi.com/downloads/) to use Gurobi.
  Academics may request a free licence from the Gurobi
  website [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Place the Gurobi licence file in your
  *home directory* or
  another [recommended location](https://support.gurobi.com/hc/en-us/articles/360013417211-Where-do-I-place-the-Gurobi-license-file-gurobi-lic).
- **ImportLDraw:** Rendering brick visualizations requires ImportLDraw, provided as a Git submodule. Follow these instructions to install ImportLDraw:
    - Download [Git LFS](https://git-lfs.com), then run `git lfs install`.
    - Install Git submodules with `git submodule update --init`.
    - Download the [LDraw parts library](https://library.ldraw.org/library/updates/complete.zip) and
      extract it in your *home directory*:
      `(cd ~ && wget https://library.ldraw.org/library/updates/complete.zip && unzip complete.zip)`.
        - If you wish to put the LDraw parts library in a different directory, set the environment variable
          `LDRAW_LIBRARY_PATH` to the path of the `ldraw` directory: `export LDRAW_LIBRARY_PATH=path/to/ldraw`.
    - Download
      this [background exr file](https://drive.google.com/file/d/1Yux0sEqWVpXGMT9Z5J094ISfvxhH-_5K/view?usp=share_link)
      and place it in the `ImportLDraw/loadldraw` subdirectory.
## Usage

Install the Python project manager [uv](https://docs.astral.sh/uv). Then run the demo with:

```zsh
uv run gradio app.py
```
