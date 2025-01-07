import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def predict_entities(input_text):
    model_path = "./biobert-ner"  

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    label_list = [
        "O", 
        "B-DISEASE",  
        "I-DISEASE"   
    ]
    tokens = tokenizer(input_text, truncation=True, return_tensors="pt", is_split_into_words=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    predicted_labels = [label_list[pred] for pred in predictions.squeeze().tolist()]

    entities = []
    current_entity = None
    for token, label in zip(tokens, predicted_labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": label[2:], "tokens": [token]}
        elif label.startswith("I-") and current_entity:
            current_entity["tokens"].append(token)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    for entity in entities:
        entity["text"] = tokenizer.convert_tokens_to_string(entity["tokens"])
        del entity["tokens"]
    return [(entity['text'], entity['entity']) for entity in entities]
iface = gr.Interface(
    fn=predict_entities,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter text for NER", lines=5),
    outputs=gr.Dataframe(headers=["Entity", "Type"], label="Recognized Entities"),
    title="NCBI NER Inference",
    description="This model recognizes diseases and other entities from text input."
)

if __name__ == "__main__":
    iface.launch()
