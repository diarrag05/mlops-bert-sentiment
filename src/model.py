from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

def train_model(train_dataset, val_dataset):
    # Charger le modèle BERT pré-entraîné
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    # Métrique accuracy
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': (predictions == labels).mean()}

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=64,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model='accuracy'
    )

    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics  # <-- Correction ajoutée ici
    )

    print("🚀 Début de l'entraînement du modèle...")
    trainer.train()
    print("✅ Entraînement terminé avec succès.")

    model_save_path = './results/saved_model'
    trainer.save_model(model_save_path)
    print(f"✅ Modèle enregistré dans {model_save_path}")
