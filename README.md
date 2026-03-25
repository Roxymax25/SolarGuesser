# Solar Guesser


Solar Guesser ist ein Computer-Vision-Projekt zur Schaetzung der installierten PV-Leistung (kWp) auf Daechern anhand von Satellitenbildern.

![SolarGuesserDemo](https://github.com/user-attachments/assets/773afcd2-cb94-4a82-82a0-25509342ee5e)

Das Projekt kombiniert:
- Geocoding und Abruf von Satellitenbildern (Mapbox API)
- Solarpanel-Erkennung per YOLO-Segmentierungsmodell
- Umrechnung erkannter Flaeche in kWp inkl. Bandbreite und Jahresertrag
- Interaktive Streamlit-Oberflaeche mit optionaler ROI-Analyse

## Was dieses Projekt zeigt (CV-Perspektive)

- End-to-end ML-Prototyp von Datenpipeline bis UI
- Training/Fine-Tuning von YOLO-Segmentation auf Solarpanel-Daten
- Praktische Geodatenverarbeitung (Adress-Geocoding, Zoom/Resolution)
- Produktnahe Schaetzlogik mit erklaerbaren Kennzahlen

## Aktueller Stand

- App: Streamlit Frontend fuer Adresssuche oder Bild-Upload
- Modell: Nutzung eines lokal verfuegbaren besten Modells (Prioritaetslogik in detector)
- Daten: Datensaetze fuer Training/Validierung im Projekt vorhanden
- Evaluation: Trainingsartefakte und Weights lokal generiert

## Tech Stack

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV, Pillow, NumPy
- Mapbox Geocoding + Static Images API

## Schnellstart

Voraussetzungen:
- Python 3.10+
- Optional CUDA-faehige GPU fuer Training/Inference-Beschleunigung
- Mapbox Access Token

1. Repository klonen und in den Ordner wechseln.
2. Virtuelle Umgebung erstellen und aktivieren.
3. Abhaengigkeiten installieren.
4. Umgebungsvariablen setzen.
5. App starten.

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

Erstelle eine .env-Datei (siehe .env.example):

```env
MAPBOX_ACCESS_TOKEN=dein_mapbox_token
```

App starten:

```bash
streamlit run app/main.py
```

## Benutzung

1. Eingabemodus waehlen:
	 - Adresse eingeben (automatisches Geocoding + Satellitenbild)
	 - Bild hochladen
2. Optional ROI aktivieren, um den Analysebereich zu begrenzen.
3. Erkennung starten.
4. Ergebnisse lesen:
	 - erkannte Panels
	 - geschaetzte Flaeche in m2
	 - kWp-Schaetzung inkl. low/high-Bereich
	 - grobe Jahresertrags-Schaetzung

## Berechnungslogik

Die Leistungsschaetzung basiert auf:

$$
kWp = A_{m^2} \times \eta \times f
$$

mit:
- $A_{m^2}$: aus Segmentpixeln abgeleitete Flaeche
- $\eta$: angenommene Panel-Effizienz (typisch 0.15 bis 0.22)
- $f$: Fill-Faktor (Standard 0.85)

Hinweis: Es handelt sich um eine technische Naeherung, nicht um ein Gutachten.

## Training

Im Repository sind Skripte fuer unterschiedliche Trainingspfade enthalten:
- Vollstaendiges Training auf dem kompletten Datensatz
- Schnelleres Training auf 10%-Subset
- Download/Training-Pipeline fuer Roboflow-basierte Datensaetze

Beispiele:

```bash
python train_model.py
python create_subset.py
python train_model_subset.py
```

## Projektstruktur

```text
Solar_Guesser/
	app/
		main.py
		detector.py
		imagery.py
		geocoder.py
		calculator.py
	datasets/
	models/
	runs/
	train_model.py
	train_model_subset.py
	create_subset.py
	requirements.txt
	README.md
```

## Grenzen und Risiken

- Bildqualitaet, Schatten und Dachneigung beeinflussen die Genauigkeit.
- Segmentierung ist modellabhaengig und datengetrieben.
- kWp-Schaetzung nutzt Annahmen zu Effizienz und Fill-Faktor.
- Ergebnisse sind fuer Screening/Orientierung geeignet, nicht fuer finale Auslegung.

## Sicherheit und Konfiguration

- Keine API-Keys in Code einchecken.
- Zugangsdaten ausschliesslich ueber Umgebungsvariablen verwalten.
- Grosse Trainingsartefakte und Datensaetze nicht im GitHub-Repo versionieren.

## Lizenz

Dieses Projekt nutzt Ultralytics YOLO (`ultralytics`) und Ultralytics-YOLO-Modelle.
Daher wird das Repository unter der AGPL-3.0-Lizenz bereitgestellt.
Details siehe `LICENSE`.

