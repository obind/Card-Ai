import os
import argparse
import subprocess

# Basis-Pfade
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "card_model.h5")

# Funktionen für die einzelnen Schritte
def start_live_detection():
    """Starte die Live-Kartenerkennung mit dem aktuellen Modell"""
    print("\n🔍 Starte Live-Kartenerkennung ...")
    subprocess.run(["python", os.path.join(BASE_DIR, "card_ai", "live_card_detector.py")])

def capture_images():
    """Starte die Bilderfassung"""
    print("\n📸 Starte Bilderfassung ...")
    subprocess.run(["python", os.path.join(BASE_DIR, "card_ai", "capture_images.py")])

def preprocess_images():
    """Starte die Bildvorverarbeitung"""
    print("\n🛠 Starte Bildvorverarbeitung ...")
    subprocess.run(["python", os.path.join(BASE_DIR, "card_ai", "enhance_images.py")])

def train_model(dataset_choice):
    """Trainiere das Modell mit dem gewählten Datensatz"""
    print("\n🎯 Starte Training des Modells ...")
    subprocess.run(["python", os.path.join(BASE_DIR, "card_ai", "train_model.py"), "--dataset", dataset_choice])

# Argumente definieren
parser = argparse.ArgumentParser(description="Automatisierter Setup-Flow für das Kartenmodell")
parser.add_argument("--skip-capture", action="store_true", help="Überspringe die Bilderfassung")
parser.add_argument("--skip-preprocess", action="store_true", help="Überspringe die Bildvorverarbeitung")
parser.add_argument("--skip-training", action="store_true", help="Überspringe das Training")
parser.add_argument("--auto-detect", action="store_true", help="Starte direkt die Live-Kartenerkennung nach dem Training")
args = parser.parse_args()

# --- SCHRITT 1: Prüfen, ob ein Modell existiert ---
if os.path.exists(MODEL_PATH):
    print(f"\n📂 Ein trainiertes Modell wurde gefunden unter: {MODEL_PATH}")
    print("1️⃣ Live-Kartenerkennung mit dem aktuellen Modell starten")
    print("2️⃣ Neues Training starten (Bilder erfassen & trainieren)")

    start_choice = input("\nWähle eine Option (1 oder 2): ").strip()

    if start_choice == "1":
        start_live_detection()
        exit(0)
    elif start_choice != "2":
        print("❌ Ungültige Eingabe. Skript wird beendet.")
        exit(1)
else:
    print("⚠️ Kein trainiertes Modell gefunden. Starte mit Datenerfassung & Training.")

# --- SCHRITT 2: Datensatz wählen ---
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

# Prüfen, ob der gewählte Dataset-Ordner existiert
if not os.path.exists(DATASET_DIR):
    print(f"⚠️ Der Dataset-Ordner '{DATASET_DIR}' wurde nicht gefunden. Möchtest du Bilder aufnehmen?")
    choice = input("Gib 'y' für Ja oder 'n' für Nein ein: ").strip().lower()
    if choice == "y":
        capture_images()
    else:
        print("❌ Ohne Bilder kann das Modell nicht trainiert werden. Skript wird beendet.")
        exit(1)

# --- SCHRITT 3: Bilderfassung ---
if not args.skip_capture:
    print("\n❓ Hast du bereits Bilder aufgenommen?")
    capture_choice = input("Gib 'y' ein, falls du diesen Schritt überspringen möchtest: ").strip().lower()
    if capture_choice != "y":
        capture_images()

# --- SCHRITT 4: Bildvorverarbeitung (falls nötig) ---
if not args.skip_preprocess and dataset_choice == "raw":
    print("\n❓ Möchtest du die Bilder vorverarbeiten, um bessere Qualität zu erhalten?")
    preprocess_choice = input("Gib 'y' ein, falls du vorverarbeiten möchtest: ").strip().lower()
    if preprocess_choice == "y":
        preprocess_images()

# --- SCHRITT 5: Modell trainieren ---
if not args.skip_training:
    print("\n🚀 Starte Modelltraining ...")
    train_model(dataset_choice)

# --- SCHRITT 6: Prüfen, ob Modell nach Training existiert ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ Kein trainiertes Modell gefunden unter: {MODEL_PATH}. Training wurde eventuell abgebrochen.")
    exit(1)

print(f"✅ Aktuelles Modell gefunden unter: {MODEL_PATH}")

# --- SCHRITT 7: Live-Kartenerkennung starten ---
print("\n❓ Möchtest du die Live-Kartenerkennung jetzt starten?")
detect_choice = input("Gib 'y' ein, falls du das Modell testen möchtest: ").strip().lower()
if detect_choice == "y" or args.auto_detect:
    start_live_detection()

print("\n✅ Alle Schritte abgeschlossen!")
