import os
import cv2

# Hauptordner der Karten
base_dir = "cards_dataset"

# Farben und Werte
suits = ["hearts", "diamonds", "spades", "clubs"]
values = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "jack", "queen", "king", "ace"
]

# Kamera initialisieren
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

# Bilderanzahl pro Karte
images_per_card = 20

# Alle Karten iterieren
for suit in suits:
    for value in values:
        card_name = f"{suit}_{value}"
        card_dir = os.path.join(base_dir, card_name)
        os.makedirs(card_dir, exist_ok=True)

        # Überprüfen, wie viele Bilder bereits vorhanden sind
        existing_images = len(os.listdir(card_dir))
        if existing_images >= images_per_card:
            print(f"Karte {card_name} bereits vollständig.")
            continue

        print(f"Fotografiere jetzt: {card_name}")
        print(f"Bilder benötigt: {images_per_card - existing_images}")

        # Bilder aufnehmen
        count = existing_images
        while count < images_per_card:
            ret, frame = cap.read()
            if not ret:
                print("Fehler beim Lesen der Kamera.")
                break

            # Zeige den Live-Feed
            cv2.imshow("Live Feed", frame)

            # 's' drücken, um ein Bild zu speichern
            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_path = os.path.join(card_dir, f"img{count + 1}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Bild gespeichert: {img_path}")
                count += 1

            # 'q' drücken, um das Programm zu beenden
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Beenden...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

print("Alle Karten fotografiert!")
cap.release()
cv2.destroyAllWindows()