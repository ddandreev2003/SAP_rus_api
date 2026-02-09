import os
import torch
import argparse
from pathlib import Path
from SAP_pipeline_flux import SapFlux
from llm_interface.llm_SAP import LLM_SAP
BASE_FOLDER = os.getcwd()

################################
API_KEY = "YOUR_API_KEY"
################################

def parse_input_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=1024, help="define the generated image height")
    parser.add_argument('--width', type=int, default=1024, help="define the generated image width")
    parser.add_argument('--seeds_list', nargs='+', type=int, default=[30498], help="define the list of seeds for the prompt generated images")
    parser.add_argument('--prompt', type=str, default="A bear is performing a handstand in the park")
    parser.add_argument('--llm', type=str, default="GPT", help="define the llm to be used, support GPT and Zephyr")
    parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face token for gated model access (or set HF_TOKEN env var)")
    parser.add_argument('--flux_version', type=str, default="1", choices=["1", "2"], help="Flux model version: 1 (default) or 2 (Flux2.0-dev)")
    args = parser.parse_args()
    return args

def load_model(hf_token=None):
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        print("Error: Hugging Face token required for gated model. Pass via --hf_token or set HF_TOKEN environment variable.")
        exit(1)
    import importlib
    import torch
    flux_version = getattr(args, 'flux_version', '1') if 'args' in globals() else '1'
    if flux_version == "2":
        Flux2Pipeline = None
        try:
            Flux2Pipeline = importlib.import_module("diffusers").Flux2Pipeline
        except Exception:
            print("Error: diffusers.Flux2Pipeline not found. Please update diffusers package.")
            exit(1)
        repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
        device = "cuda:0"
        torch_dtype = torch.bfloat16
        model = Flux2Pipeline.from_pretrained(
            repo_id, torch_dtype=torch_dtype, token=hf_token
        )
        model.to(device, dtype=torch_dtype)
        return model
    else:
        model = SapFlux.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        model.enable_model_cpu_offload()
        return model

def save_results(images, prompt, seeds_list):
    prompt_model_path = os.path.join(BASE_FOLDER, "results", prompt)
    Path(prompt_model_path).mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(seeds_list):
        images[i].save(os.path.join(prompt_model_path, f"Seed{seed}.png"))

def generate_models_params(args, SAP_prompts):
    generators_lst = []
    for seed in args.seeds_list:
        generator = torch.Generator()
        generator.manual_seed(seed)
        generators_lst.append(generator)
    params = {"height": args.height, 
              "width": args.width,
              "num_inference_steps": 50,
              "generator": generators_lst,
              "num_images_per_prompt": len(generators_lst),
              "guidance_scale": 3.5, 
              "sap_prompts": SAP_prompts}
    return params

def run(args):
    # Load SAP prompts from ContraBench_prompt_mapping.json
    import json
    mapping_path = os.path.join(BASE_FOLDER, "benchmarks", "SAP_prompts", "ContraBench_prompt_mapping.json")
    with open(mapping_path, 'r') as f:
        prompt_mapping = json.load(f)
    if args.prompt not in prompt_mapping:
        print(f"Error: Prompt '{args.prompt}' not found in ContraBench_prompt_mapping.json.")
        exit(1)
    SAP_prompts = prompt_mapping[args.prompt]
    params = generate_models_params(args, SAP_prompts)
    # Load model
    model = load_model(hf_token=args.hf_token)
    # Run model
    images = model(**params).images
    # Save results
    save_results(images, args.prompt, args.seeds_list)

def main():
    args = parse_input_arguments()
    # pass update args with defualts
    run(args)
    
if __name__ == "__main__":
    main()