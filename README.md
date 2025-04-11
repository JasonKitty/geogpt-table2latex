---
license: mit
language:
- en
- zh
base_model:
- OpenGVLab/InternVL2-1B
pipeline_tag: image-text-to-text
tags:
- table-understanding
- latex
- image-to-text
- academic-document
---

# 🧮 GeoGPT-Table2LaTeX
[🚀 Quick Start](#quick-start) • [🧠 Model Weights](#model-weights) • [🌐 Web UI Demo](#web-ui-demo-local) • [📊 Dataset Highlights](#training-dataset-highlights) • [🙌 Citation](#citation)

GeoGPT-Table2LaTeX is an end-to-end LaTeX table recognition tool specialized for academic documents. Fine-tuned on over 2 million high-quality table samples and trained with 32×A100 GPUs, it converts table images into structured, compilable LaTeX code — and comes with an interactive, offline-friendly web interface.

<p align="left">
  <img src="table2latex_web_demo/assets/demo.gif" alt="GeoGPT Table2LaTeX Demo" width="480"/>
</p>

- 📸 **Image-to-LaTeX model**, fine-tuned from [InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B) trained on clean, source-rendered supervision
- 💻 **Web-based interface** for uploading table images and getting structured LaTeX code, with optional rendering preview
- 🧰 **Lightweight & offline-friendly**, ideal for integration or local deployment

To further optimize recognition accuracy, formatting quality, and LaTeX compilability, we apply **Generalized Reward Policy Optimization (GRPO)** — with rewards from both image similarity and structure matching.

<h2 id="training-dataset-highlights">📊 Training Dataset Highlights</h2>

We built a **large-scale, high-quality training dataset** for table recognition based on a **compilation-based strategy**, ensuring precise alignment between table images and their LaTeX representations.

Our dataset includes:

- ✅ **Over 1.2 million high-quality English samples**, fully cleaned and structurally normalized
- ✅ **Over 0.8 million Chinese samples**, translated and cleaned with assistance of [Qwen-VL-72B](https://huggingface.co/Qwen/Qwen-VL-Chat)

Although the dataset is **not yet open-sourced**, we plan to release it in the future.

Key features include:

- 📄 **Source-aligned**: Tables are extracted directly from LaTeX source files, avoiding noisy PDF parsing.
- ⚙️ **Dependency-aware rendering**: Automatically detects and includes required LaTeX packages for accurate compilation.
- 🧹 **Cleaned & normalized LaTeX**: Formatting and grammar inconsistencies are improved for better model learning.

<h2 id="model-weights">🧠 Model Weights</h2>

The model weights are available on the 🔗 [Hugging Face Hub](https://huggingface.co/JasonKitty/geogpt-table2latex-weights)  

<h2 id="quick-start">🚀 Quick Start</h2>

### 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://huggingface.co/JasonKitty/geogpt-table2latex
cd geogpt-table2latex
```

To use this project, you need:

- A working **GPU environment with CUDA installed**
- A properly installed **Transformers environment**, including `torch`, `transformers`, and `accelerate`

Once your environment is ready, you can install this project's requirements:

```bash
pip install -r requirements.txt
```

Make sure the model weights are located under `weights/InternVL2-1B-finetuned-table-v5/`.

For `lmdeploy`, install following their instructions:
👉 https://github.com/InternLM/lmdeploy

### 🔹 Transformers Inference

```bash
python table2latex_transformers.py \
  --image_path weights/InternVL2-1B-finetuned-table-v5/examples/2.jpg \
  --model_path weights/InternVL2-1B-finetuned-table-v5
```

### 🔹 lmdeploy Fast Inference

```bash
python table2latex_lmdeploy.py \
  --image weights/InternVL2-1B-finetuned-table-v5/examples/2.jpg \
  --model weights/InternVL2-1B-finetuned-table-v5
```

<h2 id="web-ui-demo-local">🌐 Web UI Demo (Local)</h2>

You can also run a local web interface to interactively upload table images and get LaTeX output + rendering.

### 📦 Step-by-step Instructions

> Requires: `lmdeploy` and GPU environment already installed.

1. **Install system dependencies** (for LaTeX rendering and image conversion):

```bash
sudo apt-get update && sudo apt-get install -y \
    texlive-full \
    poppler-utils \
    libgl1 \
    curl
```

2. **Install frontend Python dependencies**:

```bash
cd table2latex_web_demo
pip install -r requirements.txt
```

3. **Launch the Web UI**:

```bash
python app.py
```

Access the interface at:  
👉 [http://localhost:8396](http://localhost:8396)

### 🖼️ Features

- 🖼️ Upload, paste, or drag & drop table images
- ⚙️ One-click conversion to LaTeX
- 🔍 Preview rendered LaTeX table using `xelatex`
- ✅ Copy code, clear session, rerun easily

> 📁 The web app uses local model via `lmdeploy` for fast inference.

## 📁 File Structure

```
.
├── table2latex_transformers.py     # Transformers-based inference
├── table2latex_lmdeploy.py         # lmdeploy inference
├── requirements.txt                # Dependencies
├── table2latex_web_demo/           # Flask-based Web UI
└── weights/
    └── InternVL2-1B-finetuned-v5-38555/
        ├── model.safetensors + config files
        └── examples/                # Demo images
```


<h2 id="citation">🙌 Citation</h2>

If you find this work helpful, please consider citing the base model and this repository.
```
@misc{geogpt_table2latex,
  title        = {GeoGPT-Table2LaTeX: Converting Table Images to LaTeX},
  author       = {{GeoGPT Team}},
  year         = {2025},
  url          = {https://huggingface.co/JasonKitty/geogpt-table2latex},
  note         = {Available at Hugging Face}
}
```