from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset

# Initialiser le tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Fonction de prétraitement des données
def preprocess_data(data: pd.DataFrame):
    # Nettoyage de texte
    data['text'] = data['text'].str.lower()
    data['text'] = data['text'].str.replace(r'[^a-z0-9 ]', '', regex=True)

    # Division des données en ensembles d'entraînement et de validation
    train_df, val_df = train_test_split(data, test_size=0.2)

    # Tokenisation
    train_encodings = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=512)
    val_encodings = tokenizer(val_df['text'].tolist(), padding=True, truncation=True, max_length=512)

    # Création des datasets PyTorch
    train_dataset = SentimentDataset(train_encodings, train_df['label'].tolist())
    val_dataset = SentimentDataset(val_encodings, val_df['label'].tolist())

    print(f"✅ Données prétraitées : {len(train_dataset)} entraînement, {len(val_dataset)} validation")
    
    return train_dataset, val_dataset