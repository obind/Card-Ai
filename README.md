# Card Recognition AI 🃏  
Ein **Custom CNN**, das Spielkarten anhand visueller Merkmale erkennt und klassifiziert.  

## 📌 Setup & Installation  

````bash
# Voraussetzungen installieren
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn h5py
````

## 📂 Daten vorbereiten  
````bash
Bilder sammeln:  
# - Speichere Kartenbilder unter raw_dataset/<Kartenname>/ (z. B. raw_dataset/hearts_2/)
# - Mindestens 50 Bilder pro Karte, verschiedene Winkel & Beleuchtungen

Vorverarbeitung starten:
python preprocess.py
# Dies erstellt processed_dataset/ mit skalierten & optimierten Bildern.
````

## 🚀 Modell trainieren  
````bash
python train.py
# Läuft für 50+ Epochen (kann angepasst werden)
# Trainiert mit TensorFlow/Keras auf einem CNN-Modell
# Das fertige Modell wird als heart_card_classifier_advanced.h5 gespeichert.
````

## 🎥 Live-Kartenerkennung  
````bash
python live_feed.py
# Erkennt Karten per Webcam & zeigt das Ergebnis in Echtzeit an.
# Falls Karten falsch erkannt werden, sollten mehr Trainingsdaten hinzugefügt werden.
````

## 📁 Dateistruktur  

Das Projekt sollte diese Struktur haben:  

```
card_counter/
│── raw_dataset/            # Unverarbeitete Bilder
│── processed_dataset/      # Vorverarbeitete Bilder (automatisch erstellt)
│── models/                 # Hier wird das trainierte Modell gespeichert
│   ├── heart_card_classifier_advanced.h5
│── train.py                # Trainiert das Modell
│── preprocess.py           # Skaliert und verarbeitet Bilder
│── live_feed.py            # Live-Kamera-Feed mit Kartenerkennung
│── README.md               # Diese Anleitung
│── .gitignore              # Verhindert das Hochladen großer Dateien
```

## 🛠 Relativer Pfad für das Modell  
Um sicherzustellen, dass das Skript unabhängig vom absoluten Dateisystem funktioniert, wurde der **relative Pfad** verwendet:  

**In `live_feed.py`:**
````python
import os
from tensorflow.keras.models import load_model

model_path = os.path.join(os.getcwd(), "models", "heart_card_classifier_advanced.h5")
model = load_model(model_path)
````

## 🔧 Anpassungen & Optimierungen  
````bash
# Epochen erhöhen: In train.py kann epochs=100 gesetzt werden für längeres Training.
# Daten augmentieren: Mehr Variationen in ImageDataGenerator() hinzufügen.
# Modell verbessern: Mehr CNN-Schichten oder Transfer Learning mit MobileNetV2 ausprobieren.
````

## ❓ FAQ & Fehlerbehebung  
````bash
# 1️⃣ Bekomme Fehler FileNotFoundError für die .h5-Datei?
# - Stelle sicher, dass sich heart_card_classifier_advanced.h5 im models/ Ordner befindet.
# - Falls nicht, trainiere das Modell erneut mit: 
python train.py

# 2️⃣ Bekomme Fehler FileNotFoundError für processed_dataset/?
# - Der Ordner wurde noch nicht erstellt. Führe die Vorverarbeitung aus:
python preprocess.py

# 3️⃣ Vorhersagen sind ungenau?
# - Trainiere mit mehr Bildern oder verbessere die Augmentation in train.py.
# - Füge verschiedene Winkel und Beleuchtungen zu den Trainingsbildern hinzu.

# 4️⃣ Bekomme ModuleNotFoundError für TensorFlow oder Keras?
pip install -r requirements.txt
````

---

**🎯 Viel Erfolg mit der Card AI! 🚀**