# Card Recognition AI 🃏
Ein **Custom CNN**, das Spielkarten anhand visueller Merkmale erkennt und klassifiziert.

## 📌 Setup & Installation

### 1. Voraussetzungen
Bevor du startest, stelle sicher, dass du folgende Abhängigkeiten installiert hast:

#### Manuelle Installation der wichtigsten Pakete:
```bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn
```

#### Nutzung einer virtuellen Umgebung:
```bash
python -m venv venv  
source venv/bin/activate  # (Mac/Linux)  
venv\Scripts\activate  # (Windows)
pip install -r requirements.txt
```
Falls du TensorFlow nicht in der virtuellen Umgebung installiert hast, stelle sicher, dass es richtig installiert ist:
```bash
pip install tensorflow-macos  # Falls du einen Mac mit M1/M2 benutzt
pip install tensorflow  # Für andere Systeme
```
Überprüfe, ob TensorFlow richtig installiert ist:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```


## 2. Daten vorbereiten
### 📸 Bilder sammeln:
- Speichere Kartenbilder unter `raw_dataset/<Kartenname>/` (z. B. `raw_dataset/hearts_2/`).
- Mindestens **50 Bilder pro Karte**, verschiedene Winkel & Beleuchtungen.


## 📖 Schreibweise der Kartennamen
Um Kompatibilitätsprobleme zwischen den Skripten zu vermeiden, müssen Kartennamen einheitlich benannt werden. Verwende die folgende Schreibweise:

- **Farben (Suits)**: `hearts`, `diamonds`, `spades`, `clubs`
- **Werte (Values)**:
  - Zahlenkarten: `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`
  - Bildkarten: `jack`, `queen`, `king`, `ace`
- **Formatierung**: `<Farbe>_<Wert>` (z. B. `hearts_2`, `spades_king`)
- **Keine Großbuchstaben**: Nur Kleinbuchstaben erlaubt (`hearts_10` statt `Hearts_10`).
- **Kein Leerzeichen, kein Sonderzeichen**: `_` als Trennzeichen nutzen.

- **Nutze das Skript `capture_images.py`, um den Prozess zu vereinfachen**:
  ```bash
  python capture_images.py
  ```
  Dies ermöglicht das einfache Erfassen von Bildern direkt über die Kamera.
  

### 🔄 Vorverarbeitung starten:
```bash
python preprocess.py
```
Dies erstellt `processed_dataset/` mit skalierten & optimierten Bildern.

## 3. Modell trainieren
```bash
python train_model.py
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

🎯 **Fertig! Dein Modell kann nun Karten erkennen!**
