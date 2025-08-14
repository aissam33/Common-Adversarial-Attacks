import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from io import BytesIO
from werkzeug.utils import secure_filename

# --- Initialisation Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Charger le modèle une seule fois au démarrage ---
model_path = "/Users/aissamhamida/Desktop/IA/penumonia_flask/chest_xray.h5"
model = load_model(model_path)

# Déterminer la taille d'entrée du modèle
_, img_height, img_width, img_channels = model.input_shape

# --- Route principale pour afficher le formulaire ---
@app.route('/')
def home():
    return render_template('upload.html')

# --- Route pour prédire ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Aucune image fournie", 400

    file = request.files['image']
    if file.filename == '':
        return "Nom de fichier invalide", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Charger l'image pour le modèle
    color_mode = "rgb" if img_channels == 3 else "grayscale"
    img = load_img(filepath, target_size=(img_height, img_width), color_mode=color_mode)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Faire la prédiction
    pred = model.predict(x)[0][0]
    # Inversion si nécessaire selon le modèle
    result = "Normal" if pred >= 0.5 else "Pneumonia"

    # Retourner la page avec image et résultat
    return render_template('result.html', filename=filename, result=result, score=f"{pred:.3f}")

# --- Route pour afficher l'image ---
@app.route('/uploads/<filename>')
def send_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# --- Lancer l'application ---
if __name__ == "__main__":
    app.run(debug=True)
