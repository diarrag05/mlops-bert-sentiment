import unittest
from src.model import train_model
from src.data_processing import preprocess_data
from src.data_extraction import load_data

class TestModel(unittest.TestCase):

    def test_train_model(self):
        data = load_data('dataset.csv')
        _, train_data, val_data = preprocess_data(data)
        try:
            train_model(train_data, val_data)
            success = True
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle : {e}")
            success = False
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
