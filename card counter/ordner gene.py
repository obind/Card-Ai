import os

# Hauptordner
base_dir = "cards_dataset"

# Farben und Werte
suits = ["hearts", "diamonds", "spades", "clubs"]
values = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "jack", "queen", "king", "ace"
]

# Ordner erstellen
os.makedirs(base_dir, exist_ok=True)
for suit in suits:
    for value in values:
        folder_name = f"{suit}_{value}"
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Ordner erstellt: {folder_path}")

print("Alle Ordner wurden erfolgreich erstellt!")