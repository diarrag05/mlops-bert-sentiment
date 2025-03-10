from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Chemin absolu pour être sûr (à adapter à ton PC)
model_path = r"C:\Users\mdiia\OneDrive\Bureau\AIVANCITY\Cours\PGE3\MLOps\project\results\checkpoint-32"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax().item()

    sentiment_labels = ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
    return sentiment_labels[predicted_class]

if __name__ == "__main__":
    exemple_texte = "This app is wonderful, I love using it every day!"
    prediction = predict(exemple_texte)
    print(f"Texte : '{exemple_texte}'")
    print(f"Sentiment prédit : {prediction}")
