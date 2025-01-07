from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_from_disk
import torch

if __name__ == '__main__':
    
    tokenized_datasets = load_from_disk('./tokenized_datasets')
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    label_list = tokenized_datasets['train'].features['ner_tags'].feature.names
    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    
    model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        dataloader_num_workers = 4,
        fp16 = True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer.train()
    model.save_pretrained("./biobert-ner")
    tokenizer.save_pretrained("./biobert-ner")

    

