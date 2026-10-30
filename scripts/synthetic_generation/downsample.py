import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path

DATASET_RESOLUTIONS = {
    "mnist": (28, 28),
    "svhn": (32, 32),
    "eurosat": (64, 64),
    "dtd": (300, 300),
    "gtsrb": (48, 48),
    "resisc45": (256, 256),
    "stanford_cars": (224, 224),
    "sun397": (224, 224)
}

DATASET_COLOR_MODES = {
    "mnist": "L",
    "svhn": "RGB",
    "eurosat": "RGB",
    "dtd": "RGB",
    "gtsrb": "RGB",
    "resisc45": "RGB",
    "stanford_cars": "RGB",
    "sun397": "RGB"
}

INPUT_DIR = "synthetic_images"
OUTPUT_DIR = "synthetic_images_downsampled"


def downsample_image(input_path, output_path, target_size, color_mode):
    try:
        img = Image.open(input_path)
        
        if color_mode == "L":
            img = img.convert("L")
        elif color_mode == "RGB":
            img = img.convert("RGB")
        
        img_downsampled = img.resize(target_size, Image.Resampling.LANCZOS)
        
        img_downsampled.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def downsample_dataset(dataset_name):
    
    input_dataset_dir = os.path.join(INPUT_DIR, dataset_name)
    output_dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    
    if not os.path.exists(input_dataset_dir):
        print(f"⚠ Dataset {dataset_name} not found, skipping")
        return
    
    target_size = DATASET_RESOLUTIONS.get(dataset_name, (224, 224))
    color_mode = DATASET_COLOR_MODES.get(dataset_name, "RGB")
    
    os.makedirs(output_dataset_dir, exist_ok=True)
    
    image_files = list(Path(input_dataset_dir).glob("*.png"))
    
    if not image_files:
        print(f"⚠ No images found in {input_dataset_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Downsampling: {dataset_name}")
    print("  Source size: 512x512")
    print(f"  Target size: {target_size[0]}x{target_size[1]}")
    print(f"  Color mode: {color_mode}")
    print(f"  Images: {len(image_files)}")
    print(f"{'='*80}")
    
    success_count = 0
    for img_path in tqdm(image_files, desc=f"  Processing {dataset_name}"):
        output_path = os.path.join(output_dataset_dir, img_path.name)
        
        if downsample_image(str(img_path), output_path, target_size, color_mode):
            success_count += 1
    
    print(f"✓ Completed: {success_count}/{len(image_files)} images downsampled")


def main():
    
    print("="*80)
    print("Downsampling Synthetic Images to Original Dataset Resolutions")
    print("="*80)
    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ Error: {INPUT_DIR} directory not found!")
        return
    
    datasets = [d for d in os.listdir(INPUT_DIR) 
                if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    if not datasets:
        print(f"\n❌ No datasets found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(datasets)} datasets to process:")
    for dataset in sorted(datasets):
        size = DATASET_RESOLUTIONS.get(dataset, (224, 224))
        mode = DATASET_COLOR_MODES.get(dataset, "RGB")
        print(f"  - {dataset:20s} → {size[0]}x{size[1]} {mode}")
    
    total_processed = 0
    for dataset in sorted(datasets):
        downsample_dataset(dataset)
        
        output_dir = os.path.join(OUTPUT_DIR, dataset)
        if os.path.exists(output_dir):
            count = len(list(Path(output_dir).glob("*.png")))
            total_processed += count
    
    print("\n" + "="*80)
    print("✓ Downsampling Complete!")
    print(f"  Total images processed: {total_processed}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    print("\nDisk usage comparison:")
    import subprocess
    try:
        original_size = subprocess.check_output(['du', '-sh', INPUT_DIR]).decode().split()[0]
        downsampled_size = subprocess.check_output(['du', '-sh', OUTPUT_DIR]).decode().split()[0]
        print(f"  Original (512x512): {original_size}")
        print(f"  Downsampled:        {downsampled_size}")
    except:
        pass


if __name__ == "__main__":
    main()
