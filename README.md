# Pipeline d'Analyse de Sentiment avec BERT

# MLOps - Analyse de Sentiment avec BERT

Ce projet implémente un pipeline complet d'analyse de sentiment basé sur **BERT**, permettant de classifier les avis clients en sentiments positifs, neutres ou négatifs.

## 📌 Fonctionnalités
- Extraction et prétraitement des données (nettoyage, tokenization)
- Entraînement d'un modèle BERT avec **Hugging Face Transformers**
- Inférence pour la prédiction des sentiments
- Tests unitaires pour garantir la robustesse du pipeline
- Gestion de version avec **Git & GitHub**


## Objectifs du projet
Les objectifs principaux du projet sont les suivants :
•	Concevoir un pipeline de traitement des données efficace pour structurer les avis clients
•	Implémenter un modèle de classification des sentiments basé sur BERT
•	Développer un système robuste d’inférence pour permettre une analyse en temps réel
•	Mettre en place des tests unitaires et une validation rigoureuse pour garantir la fiabilité du modèle
•	Assurer une collaboration fluide via Git et GitHub pour le suivi et la gestion des contributions

## Paramètres et Entraînement du Modèle
L’entraînement du modèle a été réalisé à partir du modèle pré-entraîné bert-base-uncased de Hugging Face, optimisé pour la classification de texte.

Les hyperparamètres suivants ont été ajustés pour améliorer la performance :
•	Batch size : 8 (pour gérer l’équilibre entre performance et ressources)
•	Learning rate : 2e-5 (valeur optimale pour éviter le surajustement)
•	Nombre d’époques : 3 (permettant un bon compromis entre entraînement et généralisation)
•	Fonction de perte : CrossEntropyLoss

Le modèle a été entraîné à l’aide de la bibliothèque Transformers de Hugging Face, avec une validation en fin d’époque pour suivre l’évolution des performances.


