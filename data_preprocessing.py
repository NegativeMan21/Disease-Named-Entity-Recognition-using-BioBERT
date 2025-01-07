from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForTokenClassification


tokenized_datasets = load_from_disk('./tokenized_datasets')
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")  

label_list = tokenized_datasets['train'].features['ner_tags'].feature.names
data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

