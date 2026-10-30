import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import json
from tqdm import tqdm
import random
from classes import DATASETS, get_prompt_for_class

IMAGES_PER_DATASET = 500
OUTPUT_BASE_DIR = "synthetic_images"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512


def setup_pipeline():
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled for memory efficiency")
        except:
            print("⚠ xformers not available")
    
    return pipe


def generate_images_for_dataset(pipe, dataset_name, dataset_info):
    
    print(f"\n{'='*80}")
    print(f"Generating images for: {dataset_info['name']}")
    print(f"Number of classes: {dataset_info['num_classes']}")
    print(f"{'='*80}\n")
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if 'classes' in dataset_info:
        classes = dataset_info['classes']
    elif 'sample_classes' in dataset_info:
        classes = dataset_info['sample_classes']
        print(f"⚠ Using {len(classes)} sample classes (full dataset has {dataset_info['num_classes']} classes)")
    else:
        print(f"⚠ No class information available for {dataset_name}")
        return
    
    num_images = IMAGES_PER_DATASET
    print(f"Generating {num_images} images by randomly sampling from {len(classes)} classes")
    
    metadata = {
        "dataset_name": dataset_name,
        "full_name": dataset_info['name'],
        "num_classes": len(classes),
        "total_images": num_images,
        "model": MODEL_ID,
        "sampling_strategy": "random_class_sampling",
        "images": []
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating images...")
    for img_idx in tqdm(range(num_images), desc=f"  {dataset_info['name']}"):
        class_name = random.choice(classes)
        
        prompt = get_prompt_for_class(dataset_name, class_name)
        
        generator = torch.Generator(device=DEVICE).manual_seed(SEED + img_idx)
        
        try:
            image = pipe(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE
            ).images[0]
            
            filename = f"{dataset_name}_{img_idx:05d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            metadata["images"].append({
                "filename": filename,
                "image_idx": img_idx,
                "class_name": class_name,
                "prompt": prompt,
                "seed": SEED + img_idx
            })
            
        except Exception as e:
            print(f"\n⚠ Error generating image {img_idx}: {e}")
            continue
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Completed {dataset_info['name']}")
    print(f"  Images saved to: {output_dir}")
    print(f"  Metadata saved to: {metadata_path}")


def main():
    
    print("="*80)
    print("Synthetic Image Generation for Multiple Datasets")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Model: {MODEL_ID}")
    print(f"  Device: {DEVICE}")
    print(f"  Images per dataset: {IMAGES_PER_DATASET}")
    print(f"  Output directory: {OUTPUT_BASE_DIR}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"  Guidance scale: {GUIDANCE_SCALE}")
    
    if DEVICE == "cuda":
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠ No GPU detected, using CPU (this will be slow!)")
    
    pipe = setup_pipeline()
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    datasets_to_generate = [
        "mnist",
        "svhn", 
        "eurosat",
        "dtd",
        "gtsrb",
        "resisc45",
        "stanford_cars",
        "sun397"
    ]
    
    for dataset_name in datasets_to_generate:
        if dataset_name in DATASETS:
            try:
                generate_images_for_dataset(pipe, dataset_name, DATASETS[dataset_name])
            except Exception as e:
                print(f"\n❌ Error processing {dataset_name}: {e}")
                continue
        else:
            print(f"⚠ Dataset {dataset_name} not found in configuration")
    
    print("\n" + "="*80)
    print("✓ All datasets processed!")
    print("="*80)


if __name__ == "__main__":
    main()
