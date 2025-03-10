import sys
import os

# Ajouter le dossier 'src' au chemin d'importation
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from data_extraction import load_data

# Charger et vérifier les données
data = load_data('dataset.csv')

if data is not None:
    print("✅ Données chargées avec succès ! Voici les 5 premières lignes :")
    print(data.head())
else:
    print("❌ Erreur lors du chargement des données.")
