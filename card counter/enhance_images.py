import cv2
import os
import numpy as np

# Speicherorte für Bilder
INPUT_DIR = "datasets/raw"
OUTPUT_DIR = "datasets/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Debug-Modus (0 = aus, 1 = an)
DEBUG = 1  

def log(msg):
    """ Debug-Logger für schnelle Prints """
    if DEBUG:
        print(f"[LOG] {msg}")

def is_blurry(img, threshold=100):
    """ Prüft, ob ein Bild unscharf ist, indem es die Varianz des Laplace-Filters berechnet """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < threshold:
        log(f"🚨 Unscharfes Bild erkannt (Varianz: {variance:.2f}) -> Wird verworfen")
        return True
    return False

def crop_to_card(img):
    """ Verbesserte Karten-Zuschneidung mit erweiterter Konturerkennung """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Konturen deutlicher machen
    edged = cv2.Canny(blurred, 50, 150)
    
    # Morphologische Operationen zur Konturverbesserung
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    card_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Nur viereckige Konturen (Karten)
                card_contour = approx
                max_area = area

    if card_contour is not None:
        x, y, w, h = cv2.boundingRect(card_contour)
        return img[y:y+h, x:x+w]
    
    log("⚠️ Keine Karte erkannt, benutze Originalbild!")
    return img  # Falls keine Karte erkannt wurde, bleibt das Bild unverändert

def adjust_brightness_contrast(img):
    """ Verbesserung der Helligkeit und Kontraste mit CLAHE (Adaptive Histogram Equalization) """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def sharpen_image(img):
    """ Verbesserte Schärfung mit Unsharp Masking """
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

def process_images():
    """ Geht alle Kartenordner durch und verarbeitet die Bilder """
    for card in os.listdir(INPUT_DIR):
        card_path = os.path.join(INPUT_DIR, card)
        output_card_path = os.path.join(OUTPUT_DIR, card)
        os.makedirs(output_card_path, exist_ok=True)

        for img_file in os.listdir(card_path):
            img_path = os.path.join(card_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                log(f"❌ Fehler beim Laden: {img_path}")
                continue

            # Prüfen, ob das Bild zu unscharf ist
            if is_blurry(img):
                continue  # Bild überspringen, wenn es zu unscharf ist

            # Bild verarbeiten
            img = adjust_brightness_contrast(img)
            cropped = crop_to_card(img)
            sharpened = sharpen_image(cropped)

            # Konvertieren zu Graustufen & Schärfen
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

            # Größe anpassen (Antialiasing für saubere Kanten)
            resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

            # Speichern
            output_path = os.path.join(output_card_path, img_file)
            cv2.imwrite(output_path, resized)
            log(f"✅ Verarbeitet: {output_path}")

    print("🎯 Alle Bilder wurden erfolgreich vorverarbeitet!")

if __name__ == "__main__":
    process_images()
