import unittest
import sys, os

# Chemin vers src/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', 'src')
sys.path.append(src_path)

from inference import predict

class TestInference(unittest.TestCase):

    def test_predict_positive(self):
        result = predict("I love using this app!")
        self.assertIn(result, ["Positif", "Très positif"])

    def test_predict_negative(self):
        result = predict("I hate this app.")
        self.assertIn(result, ["Négatif", "Très négatif"])

if __name__ == '__main__':
    unittest.main()
