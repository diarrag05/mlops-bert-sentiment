import sys
import os

# Ajouter le dossier 'src' au chemin d'importation
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from data_extraction import load_data
from data_processing import preprocess_data

# Charger les données
data = load_data('dataset.csv')

# Vérifier que les données existent
if data is not None:
    # Prétraitement (nettoyage + tokenisation + split train/val)
    tokens, train_data, val_data = preprocess_data(data)

    # Affichage résultats
    print("\n✅ Tokens générés (extrait):")
    print(tokens[:2])  # affiche les 2 premiers tokens uniquement à titre d'exemple

    print(f"\n✅ Taille des données d'entraînement : {len(train_data)}")
    print(f"✅ Taille des données de validation : {len(val_data)}")

    print("\n✅ Extrait des données d'entraînement :")
    print(train_data.head())
else:
    print("❌ Échec du chargement initial des données.")
