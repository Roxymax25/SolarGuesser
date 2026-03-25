"""
Solar Panel Detection Model Training Script (Subset)

Dieses Script verwendet nur 10% des ursprünglichen Datasets (ca. 3.500 Trainingsbilder).
"""

import os
import zipfile
import requests
from pathlib import Path
from ultralytics import YOLO


DATASET_DIR = Path("datasets/nl-solar-panel-seg-10percent")


def fix_data_yaml(dataset_path: str):
    """Korrigiert die Pfade in data.yaml für lokales Training."""
    data_yaml = Path(dataset_path) / "data.yaml"
    
    if not data_yaml.exists():
        return
    
    import yaml
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Absolute Pfade setzen
    dataset_path = Path(dataset_path).absolute()
    config['path'] = str(dataset_path)
    config['train'] = 'train/images'
    config['val'] = 'valid/images'
    
    with open(data_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"data.yaml aktualisiert mit lokalen Pfaden")


def train_model(dataset_path: str, epochs: int = 50, batch_size: int = 8, img_size: int = 640):
    """
    Trainiert ein YOLOv8-seg Modell auf dem Solar Panel Dataset (10% Subset).
    
    Args:
        dataset_path: Pfad zum heruntergeladenen Dataset
        epochs: Anzahl der Trainings-Epochen
        batch_size: Batch-Größe (reduziere bei wenig VRAM)
        img_size: Bildgröße für Training
    """
    
    # Korrigiere data.yaml Pfade
    fix_data_yaml(dataset_path)
    
    # YOLOv11m Segmentation Model als Basis
    model = YOLO("yolo11m-seg.pt")
    
    data_yaml = os.path.join(dataset_path, "data.yaml")
    print(f"Dataset config: {data_yaml}")
    
    # Training starten
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="solar_panel_seg_10pct",
        project="runs/segment",
        patience=15,
        save=True,
        plots=True,
        device=0,
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
    )
    
    return results

def export_model(run_path: str):
    """Exportiert das beste Modell."""
    
    best_model_path = Path(run_path) / "weights" / "best.pt"
    
    if best_model_path.exists():
        # Kopiere das beste Modell
        import shutil
        output_path = Path("models") / "solar_panel_best_10pct.pt"
        output_path.parent.mkdir(exist_ok=True)
        shutil.copy(best_model_path, output_path)
        print(f"Bestes Modell gespeichert unter: {output_path}")
        return output_path
    else:
        print(f"Kein Modell gefunden unter: {best_model_path}")
        return None

def main():
    print("=" * 60)
    print("Solar Panel Segmentation Model Training (YOLOv11m, 10% Subset)")
    print("=" * 60)
    
    # 1. Dataset checken
    print("\nPruefe Dataset...")
    if not (DATASET_DIR / "data.yaml").exists():
        print(f"Fehler: Dataset nicht gefunden: {DATASET_DIR}")
        print("Bitte zuerst ausfuehren: python create_subset.py")
        return
    
    print(f"Dataset: {DATASET_DIR}")
    
    # 2. Training starten
    print("\nStarte Training...")
    print("   (Das wird viel schneller sein als mit dem vollstaendigen Dataset)")
    print("   Du kannst den Fortschritt in Tensorboard verfolgen:")
    print("   tensorboard --logdir runs/segment")
    
    # Training Parameter
    results = train_model(
        dataset_path=str(DATASET_DIR),
        epochs=50,
        batch_size=8,
        img_size=640
    )
    
    # 3. Modell exportieren
    print("\nExportiere Modell...")
    run_path = Path("runs/segment/solar_panel_seg_10pct")
    model_path = export_model(run_path)
    
    if model_path:
        print("\n" + "=" * 60)
        print("Training abgeschlossen!")
        print(f"   Modell: {model_path}")
        print("\n   Wenn du dieses Modell in der App verwenden moechtest,")
        print("   aktualisiere app/detector.py mit dem neuen Pfad.")
        print("=" * 60)

if __name__ == "__main__":
    main()
