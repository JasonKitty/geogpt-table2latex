import argparse
import torch
import os
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms as T

# --- Image Preprocess ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(size=448):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

def preprocess_image(image_path, size=448):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(size)
    return transform(image).unsqueeze(0)

# --- Inference ---
def infer(model, tokenizer, image_tensor):
    prompt = "<image>\nConvert this table to LaTeX."
    image_tensor = image_tensor.to(torch.bfloat16).cuda()
    gen_cfg = dict(max_new_tokens=8192, do_sample=False)
    response = model.chat(tokenizer, image_tensor, prompt, generation_config=gen_cfg)
    return response

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to model dir')
    parser.add_argument('--image_path', required=True, help='Path to input image')
    args = parser.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).cuda().eval()

    image_tensor = preprocess_image(args.image_path)
    latex_code = infer(model, tokenizer, image_tensor)
    print("\n--- LaTeX Output ---\n")
    print(latex_code)
