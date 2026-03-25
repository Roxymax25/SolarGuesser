"""
Solar Panel Detection Model Training Script

Dieses Script lädt ein Dataset von Roboflow und trainiert ein YOLOv8 Segmentierungsmodell.

ANLEITUNG:
1. Gehe zu: https://universe.roboflow.com/electasolar/nl-solar-panel-seg/dataset/1
2. Klicke "Download Dataset" -> Format: "YOLOv8" -> "show download code"
3. Kopiere den API Key und füge ihn unten ein
4. Führe dieses Script aus: python train_model.py

Alternativ: Lade das Dataset manuell herunter und entpacke es nach datasets/nl-solar-panel-seg/
"""

import os
import zipfile
import requests
from pathlib import Path
from ultralytics import YOLO


DATASET_DIR = Path("datasets/nl-solar-panel-seg")


def download_dataset_manual():
    """
    Anleitung für manuellen Download.
    """
    print("\n" + "=" * 60)
    print("📥 DATASET DOWNLOAD")
    print("=" * 60)
    print("""
1. Gehe zu: https://universe.roboflow.com/electasolar/nl-solar-panel-seg/dataset/1
2. Erstelle einen kostenlosen Roboflow Account (falls nicht vorhanden)
3. Klicke "Download Dataset"
4. Wähle Format: "YOLOv8"
5. Klicke "Continue" und dann "Download zip to computer"
6. Entpacke die ZIP-Datei nach: datasets/nl-solar-panel-seg/

Die Ordnerstruktur sollte so aussehen:
datasets/
  nl-solar-panel-seg/
    data.yaml
    train/
      images/
      labels/
    valid/
      images/
      labels/
""")
    
    # Check if dataset already exists
    if (DATASET_DIR / "data.yaml").exists():
        print(f"\n✅ Dataset gefunden unter: {DATASET_DIR}")
        return str(DATASET_DIR)
    
    print(f"\n❌ Dataset nicht gefunden unter: {DATASET_DIR}")
    print("Bitte lade das Dataset manuell herunter (siehe oben).")
    
    input("\nDrücke Enter wenn das Dataset heruntergeladen ist...")
    
    if (DATASET_DIR / "data.yaml").exists():
        return str(DATASET_DIR)
    else:
        raise FileNotFoundError(f"Dataset nicht gefunden: {DATASET_DIR / 'data.yaml'}")


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
    
    print(f"✅ data.yaml aktualisiert mit lokalen Pfaden")


def train_model(dataset_path: str, epochs: int = 80, batch_size: int = 8, img_size: int = 640):
    """
    Trainiert ein YOLOv8-seg Modell auf dem Solar Panel Dataset.
    
    Args:
        dataset_path: Pfad zum heruntergeladenen Dataset
        epochs: Anzahl der Trainings-Epochen
        batch_size: Batch-Größe (reduziere bei wenig VRAM)
        img_size: Bildgröße für Training
    """
    
    # Korrigiere data.yaml Pfade
    fix_data_yaml(dataset_path)
    
    # FRISCHES Training mit vortrainiertem Modell
    print("📂 Starte FRISCHES Training mit yolo11m-seg.pt (voller Datensatz)")
    model = YOLO("yolo11m-seg.pt")
    
    data_yaml = os.path.join(dataset_path, "data.yaml")
    print(f"\n📁 Dataset config: {data_yaml}")
    
    # Training starten - VOLLER DATENSATZ
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="solar_panel_seg_full",
        project="runs/segment",
        patience=15,  # EarlyStopping nach 15 Epochen ohne Verbesserung
        # VOLLER DATENSATZ - keine fraction
        # fraction=1.0 ist default
        # Standard LR für frisches Training
        lr0=0.01,
        lrf=0.01,  # Final LR = lr0 * lrf
        warmup_epochs=3,
        save=True,
        plots=True,
        device=0,  # GPU 0
        workers=4,
        # Augmentationen für bessere Generalisierung
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,  # Volle Mosaic-Augmentation
        mixup=0.1,   # Zusätzliche Mixup-Augmentation
    )
    
    return results

def export_model(run_path: str):
    """Exportiert das beste Modell."""
    
    best_model_path = Path(run_path) / "weights" / "best.pt"
    
    if best_model_path.exists():
        # Kopiere das beste Modell
        import shutil
        output_path = Path("models") / "solar_panel_best.pt"
        output_path.parent.mkdir(exist_ok=True)
        shutil.copy(best_model_path, output_path)
        print(f"\n✅ Bestes Modell gespeichert unter: {output_path}")
        return output_path
    else:
        print(f"❌ Kein Modell gefunden unter: {best_model_path}")
        return None

def main():
    print("=" * 60)
    print("🔆 Solar Panel Segmentation Model Training")
    print("=" * 60)
    
    # 1. Dataset checken/herunterladen
    print("\n📥 Prüfe Dataset...")
    dataset_path = download_dataset_manual()
    print(f"✅ Dataset: {dataset_path}")
    
    # 2. Training starten
    print("\n🏋️ Starte Training mit VOLLEM Datensatz...")
    print("   ⏱️  Geschätzte Zeit: 6-12 Stunden (je nach GPU)")
    print("   📊 Du kannst den Fortschritt in Tensorboard verfolgen:")
    print("   tensorboard --logdir runs/segment")
    
    # Training Parameter - VOLLER DATENSATZ, 80 Epochen
    results = train_model(
        dataset_path=dataset_path,
        epochs=80,  # 80 Epochen für gründliches Training
        batch_size=8,  # Reduziere auf 4 bei wenig VRAM (<8GB)
        img_size=640
    )
    
    # 3. Modell exportieren
    print("\n📦 Exportiere Modell...")
    run_path = Path("runs/segment/solar_panel_seg_full")
    model_path = export_model(run_path)
    
    if model_path:
        print("\n" + "=" * 60)
        print("🎉 Training abgeschlossen!")
        print(f"   Modell: {model_path}")
        print("\n   Das Modell wird automatisch von detector.py verwendet.")
        print("   Starte die App neu: streamlit run app/main.py")
        print("=" * 60)

if __name__ == "__main__":
    main()
