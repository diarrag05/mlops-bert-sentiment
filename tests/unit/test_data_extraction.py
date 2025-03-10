import unittest
import sys
import os

# Ajouter le dossier 'src' au chemin d'importation
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(src_path)

from data_extraction import load_data
import pandas as pd

class TestDataExtraction(unittest.TestCase):

    def test_load_data_success(self):
        data = load_data('dataset.csv')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('text', data.columns)
        self.assertIn('label', data.columns)

    def test_load_data_file_not_found(self):
        data = load_data('fichier_inexistant.csv')
        self.assertIsNone(data)

    def test_load_data_empty_file(self):
        data = load_data('fichier_vide.csv')
        self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()
