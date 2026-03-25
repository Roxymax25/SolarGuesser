"""
Download dataset and train solar panel detection model.
"""
import os
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Roboflow API
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "electasolar"
PROJECT = "nl-solar-panel-seg"
VERSION = 1
FORMAT = "yolov8"

DATASET_DIR = Path("datasets/nl-solar-panel-seg")


def download_dataset():
    """Download the dataset from Roboflow."""
    if not API_KEY:
        raise ValueError(
            "ROBOFLOW_API_KEY missing. Please set it in your environment or .env file."
        )

    print("📥 Downloading NL Solar Panel Seg dataset from Roboflow...")
    
    # Roboflow download URL
    url = f"https://universe.roboflow.com/ds/{PROJECT}?key={API_KEY}"
    
    # Use Roboflow API
    download_url = f"https://app.roboflow.com/ds/{PROJECT}/{VERSION}/{FORMAT}?api_key={API_KEY}"
    
    # Alternative: Direct API call
    api_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/{FORMAT}"
    params = {"api_key": API_KEY}
    
    print(f"Requesting: {api_url}")
    
    response = requests.get(api_url, params=params)
    
    if response.status_code != 200:
        print(f"API response: {response.status_code}")
        print(response.text[:500])
        
        # Try alternative method
        print("\n🔄 Trying alternative download method...")
        return download_via_roboflow_sdk()
    
    data = response.json()
    
    if "export" in data and "link" in data["export"]:
        zip_url = data["export"]["link"]
        print(f"📦 Downloading from: {zip_url[:80]}...")
        
        # Download ZIP
        zip_response = requests.get(zip_url, stream=True)
        zip_path = DATASET_DIR.parent / "dataset.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_size = int(zip_response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in zip_response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r   Downloaded: {downloaded/1024/1024:.1f} MB ({pct:.0f}%)", end="")
        
        print(f"\n✅ Downloaded: {zip_path}")
        
        # Extract
        print("📂 Extracting...")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
        
        print(f"✅ Extracted to: {DATASET_DIR}")
        
        # Cleanup
        zip_path.unlink()
        
        return str(DATASET_DIR)
    else:
        print("❌ No download link in response")
        print(data)
        return download_via_roboflow_sdk()


def download_via_roboflow_sdk():
    """Download using the Roboflow SDK."""
    try:
        from roboflow import Roboflow
        
        print("🔧 Using Roboflow SDK...")
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION).download(FORMAT, location=str(DATASET_DIR))
        
        print(f"✅ Dataset downloaded to: {DATASET_DIR}")
        return str(DATASET_DIR)
    except Exception as e:
        print(f"❌ SDK Error: {e}")
        raise


def fix_data_yaml(dataset_path: str):
    """Fix paths in data.yaml for local training."""
    import yaml
    
    data_yaml = Path(dataset_path) / "data.yaml"
    
    if not data_yaml.exists():
        # Check subdirectories
        for sub in Path(dataset_path).iterdir():
            if sub.is_dir():
                alt_yaml = sub / "data.yaml"
                if alt_yaml.exists():
                    data_yaml = alt_yaml
                    break
    
    if not data_yaml.exists():
        print(f"⚠️ data.yaml not found in {dataset_path}")
        return None
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Original config: {config}")
    
    # Set absolute paths
    config['path'] = str(Path(dataset_path).absolute())
    config['train'] = 'train/images'
    config['val'] = 'valid/images'
    
    with open(data_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Updated data.yaml")
    return str(data_yaml)


def train_model(data_yaml: str, epochs: int = 50):
    """Train YOLOv8 segmentation model."""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("🏋️ STARTING TRAINING")
    print("=" * 60)
    
    # Load base model
    model = YOLO("yolov8s-seg.pt")
    
    print(f"📁 Data config: {data_yaml}")
    print(f"⏱️ Epochs: {epochs}")
    print(f"💾 Saving to: runs/segment/solar_panel_seg/")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,  # RTX 3070 can handle batch 16
        name="solar_panel_seg",
        project="runs/segment",
        patience=15,
        save=True,
        plots=True,
        device=0,  # Use CUDA GPU
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        workers=4,  # Use multiple workers for faster data loading
    )
    
    return results


def export_best_model():
    """Copy best model to models folder."""
    import shutil
    
    best_path = Path("runs/segment/solar_panel_seg/weights/best.pt")
    
    if best_path.exists():
        output = Path("models/solar_panel_best.pt")
        output.parent.mkdir(exist_ok=True)
        shutil.copy(best_path, output)
        print(f"\n✅ Best model saved to: {output}")
        return output
    else:
        print(f"❌ No model found at: {best_path}")
        return None


def main():
    print("=" * 60)
    print("🔆 SOLAR PANEL MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Download dataset
    dataset_path = download_dataset()
    
    # 2. Fix data.yaml
    data_yaml = fix_data_yaml(dataset_path)
    
    if data_yaml is None:
        print("❌ Could not find data.yaml. Check dataset structure.")
        return
    
    # 3. Train model
    train_model(data_yaml, epochs=50)
    
    # 4. Export best model
    export_best_model()
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("\nRestart the app to use the new model:")
    print("  .\\venv\\Scripts\\streamlit.exe run app/main.py")


if __name__ == "__main__":
    main()
