import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.image import resize
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod

# --- Entrées : chemins du modèle et de l'image ---
model_path = "/Users/aissamhamida/Downloads/Pneumonia-Detection-main/chest_xray.h5"
image_path = "/Users/aissamhamida/Downloads/Pneumonia-Detection-main/Test Images/Normal.jpeg"
output_path = "/Users/aissamhamida/Desktop/IA/image_test_adversariale1.png"

# --- 1. Charger le modèle pré-entraîné ---
model = load_model(model_path)
classifier = KerasClassifier(model=model, clip_values=(0, 1))

# --- 2. Vérifier la taille attendue par le modèle ---
input_shape = model.input_shape  # ex: (None, 128, 128, 3)
_, img_height, img_width, img_channels = input_shape

# --- 3. Charger l'image originale ---
img_original = load_img(image_path)
orig_width, orig_height = img_original.size  # Garde la taille originale
color_mode = "rgb" if img_channels == 3 else "grayscale"

# --- 4. Redimensionner pour le modèle ---
img_resized = load_img(image_path, target_size=(img_height, img_width), color_mode=color_mode)
x = img_to_array(img_resized) / 255.0
x = np.expand_dims(x, axis=0)

# --- 5. Créer l'attaque FGSM avec eps faible pour rester réaliste ---
attack = FastGradientMethod(estimator=classifier, eps=0.01)
x_adv = attack.generate(x=x)

# --- 6. Reconvertir l'image adversariale à la taille originale ---
adv_image_resized = array_to_img(x_adv[0])
adv_image = adv_image_resized.resize((orig_width, orig_height))

# --- 7. Sauvegarder l'image adversariale ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
adv_image.save(output_path)
print(f"Image adversariale générée et sauvegardée sous : {output_path}")
