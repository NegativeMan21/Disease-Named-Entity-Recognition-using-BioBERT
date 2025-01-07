from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline

dataset = load_dataset("ncbi_disease", trust_remote_code=True)

model_checkpoint = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: 
                label_ids.append(-100)  
            elif word_idx != previous_word_idx: 
                label_ids.append(label[word_idx])
            else: 
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    print(labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def clean_dataset(example):
    if not isinstance(example["tokens"], list) or not isinstance(example["ner_tags"], list):
        print("Invalid entry detected! Example removed:")
        print(example)
        return False
    if len(example["tokens"]) != len(example["ner_tags"]):
        print("Mismatch in tokens and ner_tags length! Example removed:")
        print(example)
        return False
    return True
    
dataset = dataset.filter(clean_dataset)
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

tokenized_datasets.save_to_disk("./tokenized_datasets")

print("Tokenization complete. Saved datasets to './tokenized_datasets'")
