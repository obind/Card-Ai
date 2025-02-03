import cv2
import os
import time

# Karte benennen und Speicherort angeben
card_name = input("Name der Karte (z.B. hearts_2): ")
raw_data_dir = "raw_dataset"  # Ordner für Rohbilder
os.makedirs(os.path.join(raw_data_dir, card_name), exist_ok=True)

# Kamera starten
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

print("Drücke 'q', um die Videoaufnahme zu beenden.")

frame_rate = 2  # Bilder pro Sekunde
last_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Kamerafeeds.")
        break

    # Zeige Live-Feed an
    cv2.imshow("Live Video", frame)

    # Speichere ein Bild alle `frame_rate` Sekunden
    if time.time() - last_time > 1 / frame_rate:
        frame_count += 1
        img_path = os.path.join(raw_data_dir, card_name, f"img{frame_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Bild gespeichert: {img_path}")
        last_time = time.time()

    # Beenden mit 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Programm beendet.")
        break
    # Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()

print(f"Video beendet. {frame_count} Bilder gespeichert in {os.path.join(raw_data_dir, card_name)}.")