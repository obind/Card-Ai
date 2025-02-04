import os
import argparse
import subprocess

# 🏠 Basisverzeichnis setzen
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
CARD_AI_DIR = os.path.join(BASE_DIR, "card_ai")  # Falls "card counter" umbenannt wurde

# ✅ Debug-Logger (0 = aus, 1 = an)
DEBUG = 1

def log(msg):
    """ Debug-Logger für schnelle Prints """
    if DEBUG:
        print(f"[LOG] {msg}")

# 📸 Bilderfassung starten
def capture_images():
    print("\n📸 Starte Bilderfassung ...")
    subprocess.run(["python", os.path.join(CARD_AI_DIR, "capture_images.py")])

# 🛠 Bilder vorverarbeiten
def preprocess_images():
    print("\n🛠 Starte Bildvorverarbeitung ...")
    subprocess.run(["python", os.path.join(CARD_AI_DIR, "enhance_images.py")])

# 🎯 Modell trainieren
def train_model(dataset_choice):
    print("\n🎯 Starte Training des Modells ...")
    subprocess.run(["python", os.path.join(CARD_AI_DIR, "train_model.py"), "--dataset", dataset_choice])

# 🔍 Live-Kartenerkennung starten
def start_live_detection():
    print("\n🔍 Starte Live-Kartenerkennung ...")
    subprocess.run(["python", os.path.join(CARD_AI_DIR, "live_card_detector.py")])

# 🏗️ Argumente definieren
parser = argparse.ArgumentParser(description="Automatisierter Setup-Flow für das Kartenmodell")
parser.add_argument("--skip-capture", action="store_true", help="Überspringe die Bilderfassung")
parser.add_argument("--skip-preprocess", action="store_true", help="Überspringe die Bildvorverarbeitung")
parser.add_argument("--skip-training", action="store_true", help="Überspringe das Training")
parser.add_argument("--auto-detect", action="store_true", help="Starte direkt die Live-Kartenerkennung nach dem Training")
args = parser.parse_args()

# 🗂️ **Schritt 1: Datensatz wählen**
print("\n🔍 Wähle den Datensatz für das Training:")
print("1️⃣ Rohbilder (raw_dataset)")
print("2️⃣ Vorverarbeitete Bilder (processed_dataset)")

dataset_choice = input("\nGib '1' für Rohbilder oder '2' für vorverarbeitete Bilder ein: ").strip()

if dataset_choice == "1":
    DATASET_DIR = os.path.join(DATASETS_DIR, "raw")
    dataset_choice = "raw"
    print("📂 Training mit **Rohbildern** ausgewählt.")
elif dataset_choice == "2":
    DATASET_DIR = os.path.join(DATASETS_DIR, "processed_dataset")
    dataset_choice = "processed"
    print("📂 Training mit **vorverarbeiteten Bildern** ausgewählt.")
else:
    print("❌ Ungültige Eingabe. Skript wird beendet.")
    exit(1)

# 🔎 **Überprüfen, ob der Datensatz existiert**
if not os.path.exists(DATASET_DIR):
    print(f"\n⚠️ Der Dataset-Ordner '{DATASET_DIR}' wurde nicht gefunden.")
    choice = input("📸 Möchtest du jetzt Bilder aufnehmen? (y/n): ").strip().lower()
    
    if choice == "y":
        capture_images()
    else:
        print("❌ Ohne Bilder kann das Modell nicht trainiert werden. Skript wird beendet.")
        exit(1)

# 🏗️ **Schritt 2: Bilderfassung (Falls nicht übersprungen)**
if not args.skip_capture:
    capture_choice = input("\n❓ Hast du bereits Bilder aufgenommen? (y = überspringen, n = neue Bilder aufnehmen): ").strip().lower()
    if capture_choice != "y":
        capture_images()

# 🎨 **Schritt 3: Bildvorverarbeitung (Falls nicht übersprungen & nötig)**
if not args.skip_preprocess and dataset_choice == "raw":
    preprocess_choice = input("\n❓ Möchtest du die Bilder vorverarbeiten, um bessere Qualität zu erhalten? (y/n): ").strip().lower()
    if preprocess_choice == "y":
        preprocess_images()

# 🧠 **Schritt 4: Modell trainieren (Falls nicht übersprungen)**
if not args.skip_training:
    print("\n🚀 Starte Modelltraining ...")
    train_model(dataset_choice)

# 🔍 **Schritt 5: Live-Kartenerkennung starten (Falls aktiviert)**
if args.auto_detect:
    start_live_detection()
else:
    detect_choice = input("\n❓ Möchtest du die Live-Kartenerkennung jetzt starten? (y/n): ").strip().lower()
    if detect_choice == "y":
        start_live_detection()

print("\n✅ **Alle Schritte erfolgreich abgeschlossen!** 🚀")
