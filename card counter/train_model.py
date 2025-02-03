from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import os

# Verzeichnis, wo die vorverarbeiteten Bilder liegen:
DATASET_DIR = os.getenv("DATASET_DIR", "../datasets/processed_dataset")

# Modell soll später hier gespeichert werden:
MODEL_PATH = os.getenv("DATASET_DIR", "../models/card_model.h5")

# Wie viele Epochen? (50 scheint erstmal gut, falls es zu Overfitting kommt, anpassen)
EPOCHS = 50

# Batch-Size bestimmt, wie viele Bilder gleichzeitig durch die GPU gehen.
# Sollte so groß sein, wie der Speicher es zulässt. 16 ist sicher, aber wenn GPU stark genug, kann das hoch.
BATCH_SIZE = 16  

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"❌ Der Dataset-Ordner '{DATASET_DIR}' wurde nicht gefunden!")

# --- DATEN AUGMENTIERUNG ---
# Hier bereite ich die Bilder vor, damit das Modell nicht nur exakt die gelernten Bilder erkennt,
# sondern sich an Variationen gewöhnt. Falls Bilder zu einheitlich sind, hilft das.
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Pixelwerte von 0-255 auf 0-1 normalisieren (hilft bei Training)
    validation_split=0.2,  # 80% Training, 20% Validierung
    rotation_range=45,  # Karten können leicht gedreht sein, also lernt Modell das mit
    width_shift_range=0.4,  # Mal nach links/rechts verschieben, um Perspektiven mit abzudecken
    height_shift_range=0.4,  
    zoom_range=0.5,  # Manchmal gezoomt
    shear_range=0.3,  # Verzerrungen, um realistische Kameraaufnahmen zu simulieren
    brightness_range=[0.7, 1.3],  # Helligkeit variieren, damit es nicht immer gleiche Lichtbedingungen braucht
    horizontal_flip=True  # Falls Karten mal seitenverkehrt sind (eventuell unnötig bei Karten, aber schadet nicht)
)

# --- TRAININGSDATEN LADEN ---
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(256, 256),  # Alle Bilder auf 256x256 skalieren
    color_mode="grayscale",  # Weil Modell eh nur in Graustufen trainiert wird
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# --- VALIDIERUNGSDATEN LADEN ---
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(256, 256),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# --- MODELL AUFBAUEN ---
# Das ist der Kern des CNN-Modells (Convolutional Neural Network)
# Wichtig: Relu als Aktivierungsfunktion für Convolutional Layer
# Softmax als letzte Aktivierung für Klassifizierung (weil mehrere Kartenklassen existieren)

model = Sequential()

# --- 1. Convolutional Block ---
# 32 Filter, 3x3 Kernelgröße, aktiviert mit Relu
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))  
model.add(BatchNormalization())  # Normalisiert Zwischenergebnisse, damit es stabiler trainiert
model.add(MaxPooling2D((2, 2)))  # Pooling reduziert Bildgröße, damit das Modell sich auf Hauptmerkmale fokussiert
model.add(Dropout(0.25))  # 25% der Neuronen werden zufällig deaktiviert, um Overfitting zu verhindern

# --- 2. Convolutional Block ---
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# --- 3. Convolutional Block ---
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# --- 4. Convolutional Block (größter Layer) ---
# Hier kommt richtig Power rein, 256 Filter
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))  # Höheres Dropout, weil Modell langsam Overfitting riskieren könnte

# --- Vollständig vernetzte Schicht (DENSE) ---
model.add(Flatten())  # Alle Features auf eine Zeile bringen
model.add(Dense(512, activation='relu'))  # Dichte Schicht mit 512 Neuronen
model.add(BatchNormalization())  # Normalisierung für stabileres Training
model.add(Dropout(0.5))  # Höchste Dropout-Rate

# --- OUTPUT SCHICHT ---
num_classes = len(train_generator.class_indices)  # Anzahl der Klassen automatisch bestimmen
model.add(Dense(num_classes, activation='softmax'))  # Softmax gibt Wahrscheinlichkeiten für jede Karte aus

# --- OPTIMIERUNG UND LOSS-FUNKTION ---
# Adam ist ein bewährter Optimizer, loss ist categorical_crossentropy weil mehrere Klassen existieren
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# --- CALLBACKS: AUTOMATISCHE ANPASSUNGEN ---
# Falls Validierungs-Loss stagniert, dann wird Learning Rate halbiert
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)  

# Falls Modell nach 10 Epochen nicht besser wird, Training abbrechen
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  

# --- TRAINING STARTEN ---
# Hier startet das eigentliche Training. Callback-Funktionen helfen, falls es Probleme gibt.
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping]
)

# --- MODELL SPEICHERN ---
# Falls Ordner noch nicht existiert, erstelle ihn
if not os.path.exists("models"):
    os.makedirs("models")

model.save(MODEL_PATH)
print(f"✅ Modell gespeichert unter: {MODEL_PATH}")
