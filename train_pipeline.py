import sys
import os

# Ajouter le dossier 'src' au chemin d'importation
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from data_extraction import load_data
from data_processing import preprocess_data
from model import train_model

# Chargement et prétraitement des données
data = load_data('dataset.csv')

if data is not None:
    data = data.sample(frac=0.05)  # seulement 5% des données
    train_dataset, val_dataset = preprocess_data(data)
    train_model(train_dataset, val_dataset)
else:
    print("❌ Échec lors du chargement initial des données.")
