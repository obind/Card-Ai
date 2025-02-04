# Card Recognition AI 🃏
Ein **Custom CNN**, das Spielkarten anhand visueller Merkmale erkennt und klassifiziert.

## 📌 Setup & Installation

### 1. Voraussetzungen
Bevor du startest, stelle sicher, dass du folgende Abhängigkeiten installiert hast:

#### Manuelle Installation der wichtigsten Pakete:
```bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn
```

#### Nutzung einer virtuellen Umgebung (empfohlen):
```bash
python -m venv venv  
source venv/bin/activate  # (Mac/Linux)  
venv\Scripts\activate  # (Windows)
pip install -r requirements.txt
```

## 2. Daten vorbereiten
### 🎨 Beispielbilder
Um den Einstieg zu erleichtern, wurden **Beispiel-Rohdaten** im Ordner `datasets/raw/` bereitgestellt. Diese Bilder können direkt genutzt werden, um das Modell zu trainieren. **Die vorverarbeiteten Bilder (`processed_dataset/`) wurden bewusst aus dem Commit entfernt**, um Speicherplatz zu sparen und die Übersichtlichkeit im Repository zu wahren.


## 🔄 Automatisierter Workflow
Falls du den gesamten Ablauf von Bilderfassung bis zur Modellnutzung automatisieren möchtest, kannst du `run_pipeline.py` verwenden:
```bash
python run_pipeline.py
```
Dieses Skript führt dich interaktiv durch den gesamten Prozess.


Falls du eigene Bilder aufnehmen möchtest, kannst du das Skript `capture_images.py` nutzen:
```bash
python capture_images.py
```

### 📚 Schreibweise der Kartennamen
Um Kompatibilitätsprobleme zwischen den Skripten zu vermeiden, müssen Kartennamen einheitlich benannt werden. Verwende die folgende Schreibweise:

- **Farben (Suits)**: `hearts`, `diamonds`, `spades`, `clubs`
- **Werte (Values)**:
  - Zahlenkarten: `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`
  - Bildkarten: `jack`, `queen`, `king`, `ace`
- **Formatierung**: `<Farbe>_<Wert>` (z. B. `hearts_2`, `spades_king`)
- **Nur Kleinbuchstaben**, keine Leerzeichen oder Sonderzeichen (`hearts_10`, nicht `Hearts 10`).




### 🔄 Vorverarbeitung starten:
Falls du die Bilder verbessern möchtest, kannst du `enhance_images.py` nutzen:
```bash
python enhance_images.py
```

Danach kannst du die Vorverarbeitung durchführen:
```bash
python preprocess.py
```
Dies erstellt `datasets/processed_dataset/` mit skalierten & optimierten Bildern.

## 3. Modell trainieren
Wähle, ob du mit den Rohdaten oder den vorverarbeiteten Daten trainieren möchtest:
```bash
python train_model.py --use-processed  # Falls du die optimierten Bilder nutzen möchtest
python train_model.py  # Falls du die Rohbilder nutzen möchtest
```
Das Modell wird unter `models/card_model.h5` gespeichert.

## 4. Live-Erkennung starten
Bearbeite `live_card_detector.py` und stelle sicher, dass der Modellpfad korrekt ist:

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modell und Klassen laden
model = load_model("#Pfad zum Modell")

classes = ['hearts_2', 'hearts_3', 'hearts_4', 'hearts_5', 'hearts_6',
           'hearts_7', 'hearts_8', 'hearts_9', 'hearts_10', 'hearts_jack',
           'hearts_queen', 'hearts_king', 'hearts_ace']

def detect_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
```

Dann starte die Live-Erkennung mit:
```bash
python live_card_detector.py
```

Falls Karten manuell benannt oder gespeichert werden, müssen sie dieser Struktur folgen, damit sie in den Skripten korrekt erkannt werden.



## 🏆 Über den Autor
Dieses Projekt wurde von **Kenneth Ballen Kallmann** entwickelt.  
Falls du Fragen hast oder es weiterentwickeln möchtest, kontaktiere mich gerne auf GitHub:  
[GitHub: KennethBall](https://github.com/obind)

## 🔏 Lizenz
Dieses Projekt steht unter der **MIT-Lizenz**. 

```
MIT License

Copyright (c) 2025 Kenneth Ballen Kallmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software...
```


