from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import torch

def evaluate_model(model_path, dataset_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenized_datasets = load_from_disk(dataset_path)
    
    validation_dataset = tokenized_datasets['validation']

    predictions = []
    true_labels = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    for batch in validation_dataset:
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0).to(device)
        labels = torch.tensor(batch['labels']).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()
        
        predictions.extend(np.argmax(logits, axis=-1).flatten())
        true_labels.extend(labels.cpu().numpy().flatten())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    mask = true_labels != -100
    true_labels = true_labels[mask]
    predictions = predictions[mask]


    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", cm, "\n")
    precision_confusion_matrix = cm.diagonal().sum() / cm.sum(axis=0).sum()
    recall_confusion_matrix = cm.diagonal().sum() / cm.sum(axis=1).sum()
    f1_confusion_matrix = 2 * (precision_confusion_matrix * recall_confusion_matrix) / (precision_confusion_matrix + recall_confusion_matrix)
    print(f"Calculated using confusion matrix\nPrecision: {precision_confusion_matrix}\n Recall: {recall_confusion_matrix}\n F1: {f1_confusion_matrix}\n")


    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    print("Calculated using sklearn.metrics.precision_recall_fscore_support:\n")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}\n")
    print("Predictions:", predictions)
    print("True Labels:", true_labels, "\n")
    print("Classification report:\n")
    print(classification_report(true_labels, predictions))

evaluate_model("./biobert-ner", "./tokenized_datasets")
