import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modell und Klassen laden
model = load_model("card_model.h5")
classes = ['hearts_2', 'hearts_3', 'hearts_4', 'hearts_5', 'hearts_6',
           'hearts_7', 'hearts_8', 'hearts_9', 'hearts_10', 'hearts_jack',
           'hearts_queen', 'hearts_king', 'hearts_ace']

def detect_card(frame):
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
        [256 - 1, 0],  # Zielauflösung: 256x256
        [256 - 1, 256 - 1],
        [0, 256 - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(frame, M, (256, 256))  # Zielauflösung anpassen
    return warp

# Kamera initialisieren
cap = cv2.VideoCapture(0)

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

        print("Input shape:", input_data.shape)  # Debugging

        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction)
        predicted_class = classes[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        # Vorhersage anzeigen
        cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Live-Feed anzeigen
    cv2.imshow("Live Feed", frame)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()