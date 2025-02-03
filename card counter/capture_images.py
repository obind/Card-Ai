import cv2
import os
import time

# Speicherort für Bilder
RAW_DATA_DIR = "datasets/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Debug-Modus (0 = aus, 1 = an)
DEBUG = 1  

def log(msg):
    """ Debug-Logger für schnelle Prints """
    if DEBUG:
        print(f"[LOG] {msg}")

def create_card_folder(card_name):
    """ Erstellt den Kartenordner, falls nicht vorhanden """
    path = os.path.join(RAW_DATA_DIR, card_name)
    os.makedirs(path, exist_ok=True)
    return path

def capture_from_video(card_name, frame_rate=2):
    """ Zeichnet Frames aus einem Live-Video auf und speichert sie """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    folder_path = create_card_folder(card_name)
    last_time = time.time()
    frame_count = len(os.listdir(folder_path))  # Falls schon Bilder existieren
    log(f"Starte Videoaufnahme für: {card_name} (Vorhandene Bilder: {frame_count})")

    print("Drücke 'q', um die Videoaufnahme zu beenden.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fehler beim Lesen des Kamerafeeds.")
            break

        # Zeige Live-Feed an
        cv2.imshow("Live Video", frame)

        # Alle paar Sekunden speichern
        if time.time() - last_time > 1 / frame_rate:
            frame_count += 1
            img_path = os.path.join(folder_path, f"img{frame_count}.jpg")
            cv2.imwrite(img_path, frame)
            log(f"📸 Bild gespeichert: {img_path}")
            last_time = time.time()

        # Beenden mit 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Beende Videoaufnahme...")
            break

    cap.release()
    cv2.destroyAllWindows()
    log(f"✅ Aufnahme abgeschlossen. {frame_count} Bilder gespeichert in {folder_path}.")

if __name__ == "__main__":
    card_name = input("🃏 Name der Karte (z.B. hearts_2): ").strip().lower()
    if card_name == "":
        print("❌ Kein Name eingegeben. Beende Skript.")
    else:
        capture_from_video(card_name)
