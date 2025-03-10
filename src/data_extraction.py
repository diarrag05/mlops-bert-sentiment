import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV et gère les erreurs.
    :param file_path: Chemin du fichier CSV
    :return: DataFrame contenant les colonnes 'text' et 'label'
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8')

        # Afficher les premières colonnes pour diagnostiquer
        print("Colonnes détectées :", data.columns.tolist())

        # Adapter les colonnes du fichier CSV aux attentes du modèle
        if 'content' in data.columns and 'score' in data.columns:
            data = data[['content', 'score']]
            data.columns = ['text', 'label']
            
            # Soustraire 1 pour obtenir des labels entre 0 et 4
            data['label'] = data['label'] - 1

            print(f"✅ Données chargées avec succès : {len(data)} lignes trouvées.")
            print(data.head())
            return data

        else:
            print("❌ Colonnes 'content' et 'score' non trouvées dans le fichier CSV.")
            return None

    except FileNotFoundError:
        print(f"❌ Fichier non trouvé : {file_path}")
        return None

    except pd.errors.EmptyDataError:
        print("❌ Le fichier CSV est vide.")
        return None

    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        return None