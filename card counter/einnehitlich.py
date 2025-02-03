import cv2
import os
import numpy as np

# Pfade
input_dir = "raw_dataset"
output_dir = "processed_dataset"
os.makedirs(output_dir, exist_ok=True)

# Funktion zum Zuschneiden auf die Karte
def crop_to_card(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    card_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                card_contour = approx
                max_area = area

    if card_contour is not None:
        x, y, w, h = cv2.boundingRect(card_contour)
        return img[y:y+h, x:x+w]
    return img

# Bilder vorverarbeiten
for card in os.listdir(input_dir):
    card_path = os.path.join(input_dir, card)
    output_card_path = os.path.join(output_dir, card)
    os.makedirs(output_card_path, exist_ok=True)

    for img_file in os.listdir(card_path):
        img_path = os.path.join(card_path, img_file)
        img = cv2.imread(img_path)

        # Karte zuschneiden
        cropped = crop_to_card(img)

        # Graustufen und Schärfen
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Schärfungsfilter
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Größe anpassen mit Antialiasing
        resized = cv2.resize(sharpened, (256, 256), interpolation=cv2.INTER_AREA)

        # Speichern
        output_path = os.path.join(output_card_path, img_file)
        cv2.imwrite(output_path, resized)

print("Bilder erfolgreich vorverarbeitet!")