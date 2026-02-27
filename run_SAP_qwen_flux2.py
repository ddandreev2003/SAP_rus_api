import os
from llm_interface.llm_SAP import LLM_SAP
from SAP_pipeline_flux import SapFlux
from diffusers import DiffusionPipeline
import torch

def main():
    description = input("Введите описание для SAP (например, 'A bear performing a handstand in the park'): ")
    print("\nГенерируем SAP-промпты через Qwen...")
    sap_result = LLM_SAP([description], llm='Qwen')[0]
    print("\nSAP результат:")
    print(sap_result)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Установите переменную окружения HF_TOKEN с вашим HuggingFace токеном!")

    print("\nЗагружаем Flux2...")
    from SAP_pipeline_flux import SapFlux
    model = SapFlux.from_pretrained(
            "diffusers/FLUX.2-dev-bnb-4bit",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            token=hf_token
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "generator": None,  # Можно добавить сиды
        "num_images_per_prompt": 1,
        "guidance_scale": 3.5,
        "sap_prompts": sap_result
    }
    print("\nГенерируем изображение...")
    images = model(**params).images
    out_path = "flux2_qwen_result.png"
    images[0].save(out_path)
    print(f"\nИзображение сохранено в {out_path}")

if __name__ == "__main__":
    main()
