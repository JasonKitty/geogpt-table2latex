# 推理脚本：image → LaTeX
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
from PIL import Image
import argparse

def image_to_latex(img_path: str, model_dir: str = "model") -> str:
    pipe = pipeline(model_dir, backend_config=TurbomindEngineConfig(session_len=12000))
    image = load_image(img_path)
    gen_config = GenerationConfig(
        max_new_tokens=8192,
        temperature=0.0,
        do_sample=False
    )
    response = pipe(("\nConvert this table to LaTeX.", image), gen_config=gen_config)
    return response.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to table image")
    parser.add_argument("--model", type=str, default=".", help="Path to model directory")
    args = parser.parse_args()

    result = image_to_latex(args.image, args.model)
    print("===== Output LaTeX =====")
    print(result)

