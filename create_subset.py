"""
Erstellt ein kleineres Subset aus dem Dataset.
"""
import shutil
from pathlib import Path
import random

def create_subset(source_dir: Path, target_dir: Path, subset_size: int):
    """
    Erstellt ein zufälliges Subset aus dem Source-Ordner.
    
    Args:
        source_dir: Quellordner mit Bildern
        target_dir: Zielordner für Subset
        subset_size: Anzahl der Bilder, die kopiert werden sollen
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Alle Bilder im Source-Ordner finden
    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    
    if not image_files:
        print(f"WARNUNG: Keine Bilder in {source_dir} gefunden")
        return 0
    
    # Zufälliges Sample erstellen
    subset_files = random.sample(image_files, min(subset_size, len(image_files)))
    
    # Bilder und Labels kopieren
    labels_dir = source_dir.parent / "labels"
    target_labels = target_dir.parent / "labels"
    target_labels.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for img_file in subset_files:
        # Bild kopieren
        shutil.copy(img_file, target_dir / img_file.name)
        
        # Passendes Label finden und kopieren
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, target_labels / label_file.name)
        
        copied_count += 1
    
    return copied_count

def main():
    import sys
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    
    print("=" * 60)
    print("DATASET SUBSET ERSTELLEN")
    print("=" * 60)
    
    # Pfade
    dataset_dir = Path("datasets/nl-solar-panel-seg")
    subset_dir = Path("datasets/nl-solar-panel-seg-10percent")
    
    # Subset-Größen (10% vom Original)
    train_subset_size = int(35267 * 0.1)  # ca. 3.527
    val_subset_size = int(2730 * 0.1)      # ca. 273
    
    print(f"\nOriginal Dataset:")
    print(f"  Training: {35267} Bilder")
    print(f"  Validierung: {2730} Bilder")
    
    print(f"\nSubset (10%):")
    print(f"  Training: {train_subset_size} Bilder")
    print(f"  Validierung: {val_subset_size} Bilder")
    
    # Subset erstellen
    print(f"\nErstelle Subset in: {subset_dir}")
    
    train_copied = create_subset(
        dataset_dir / "train" / "images",
        subset_dir / "train" / "images",
        train_subset_size
    )
    
    val_copied = create_subset(
        dataset_dir / "valid" / "images",
        subset_dir / "valid" / "images",
        val_subset_size
    )
    
    print(f"\nKopiert:")
    print(f"  {train_copied} Trainingsbilder")
    print(f"  {val_copied} Validierungsbilder")
    
    # data.yaml für Subset erstellen
    import yaml
    
    # Original data.yaml lesen
    with open(dataset_dir / "data.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Pfade anpassen
    config['path'] = str(subset_dir.absolute())
    config['train'] = 'train/images'
    config['val'] = 'valid/images'
    
    # Neue data.yaml speichern
    with open(subset_dir / "data.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\ndata.yaml erstellt: {subset_dir / 'data.yaml'}")
    
    print("\n" + "=" * 60)
    print("SUBSET ERSTELLT!")
    print("=" * 60)
    print(f"\nJetzt trainiere mit dem Subset:")
    print(f"  python train_model_subset.py")
    print(f"\nODER manuell Pfad in train_model.py anpassen:")
    print(f"  dataset_path = '{subset_dir}'")

if __name__ == "__main__":
    main()
