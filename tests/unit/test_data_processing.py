import unittest
from src.data_processing import preprocess_data
from src.data_extraction import load_data

class TestDataProcessing(unittest.TestCase):

    def test_preprocess_data(self):
        data = load_data('dataset.csv')
        tokens, train_data, val_data = preprocess_data(data)
        self.assertIsNotNone(tokens)
        self.assertGreater(len(train_data), 0)
        self.assertGreater(len(val_data), 0)

if __name__ == '__main__':
    unittest.main()
