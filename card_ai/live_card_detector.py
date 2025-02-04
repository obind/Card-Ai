import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Basisverzeichnis bestimmen
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Standard-Pfad für das Modell
MODEL_PATH = os.path.join(BASE_DIR, "models", "card_model.h5")

# Prüfen, ob das Modell existiert
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modell nicht gefunden unter: {MODEL_PATH} \nBitte trainiere das Modell zuerst!")

# Modell einmalig laden
print(f"📂 Lade Modell aus: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Beide Verzeichnisse laden, damit man sieht, welche Kartenklassen existieren
raw_path = os.path.join(BASE_DIR, "datasets", "raw")
processed_path = os.path.join(BASE_DIR, "datasets", "processed_dataset")

# Anzeigen, welche Kartenklassen in beiden existieren
raw_classes = sorted(os.listdir(raw_path)) if os.path.exists(raw_path) else []
processed_classes = sorted(os.listdir(processed_path)) if os.path.exists(processed_path) else []


# --- 🔍 Wähle den Datensatz ---
print("\n🔍 Wähle den Datensatz für die Klassenerkennung:")
print(f"1️⃣ Rohbilder {raw_classes}")
print(f"2️⃣ Vorverarbeitete Bilder  {processed_classes}")

dataset_choice = input("\nGib '1' für Rohbilder oder '2' für vorverarbeitete Bilder ein: ").strip()


print("\n🔍 **Gefundene Kartenklassen:**")
print(f"🟢 Rohbilder: {raw_classes}")
print(f"🔵 Vorverarbeitete Bilder: {processed_classes}")

if dataset_choice == "1":
    DATASET_PATH = raw_path
    print("📂 Nutze **Rohbilder** als Trainingsgrundlage.")
elif dataset_choice == "2":
    DATASET_PATH = processed_path
    print("📂 Nutze **vorverarbeitete Bilder** als Trainingsgrundlage.")
else:
    print("❌ Ungültige Eingabe. Skript wird beendet.")
    exit(1)

# Prüfen, ob der gewählte Datensatz existiert
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Kein Datensatz gefunden unter {DATASET_PATH} \nBitte stelle sicher, dass du Bilder aufgenommen hast.")

# Dynamisch die Kartenklassen aus dem Dataset laden
classes = sorted(os.listdir(DATASET_PATH))
print(f"🔍 **Final verwendete Klassen:** {classes}")

# --- Funktionen für Kartenerkennung ---
def detect_card(frame):
    """ Ermittelt die Kontur einer möglichen Karte im Frame """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                card_contour = approx
                max_area = area

    return card_contour

def crop_card(frame, contour):
    """ Schneidet die erkannte Karte aus und transformiert sie in die richtige Form """
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dst = np.array([
        [0, 0],
        [256 - 1, 0],
        [256 - 1, 256 - 1],
        [0, 256 - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(frame, M, (256, 256))
    return warp

# Kamera initialisieren
cap = cv2.VideoCapture(0)

print("\n🔴 **Live-Kartenerkennung gestartet!**")
print("Drücke **'q'**, um das Programm zu beenden und zurück zur Pipeline zu kehren.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Kamerafeeds.")
        break

    # Karte erkennen
    card_contour = detect_card(frame)
    if card_contour is not None:
        cv2.drawContours(frame, [card_contour], -1, (0, 255, 0), 2)
        cropped_card = crop_card(frame, card_contour)

        # Bild vorbereiten und Vorhersage
        cropped_card = cv2.cvtColor(cropped_card, cv2.COLOR_BGR2GRAY)  # Graustufen
        normalized = cropped_card / 255.0  # Normalisieren
        input_data = np.expand_dims(normalized, axis=(0, -1))  # Shape (1, 256, 256, 1)

        # **🔴 WICHTIG: Modell existiert jetzt immer korrekt!**
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction)

        # **Falls das Modell mehr Klassen hat als vorhanden, Fehler abfangen**
        if predicted_index < len(classes):
            predicted_class = classes[predicted_index]
            confidence = prediction[0][predicted_index] * 100
            cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "❌ Unbekannte Karte", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Live-Feed anzeigen
    cv2.imshow("Live Feed", frame)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 **Live-Kartenerkennung wird beendet...**")
        break

cap.release()
cv2.destroyAllWindows()
