from transformers import AutoTokenizer, AutoModelForTokenClassification

# Define model name
model_name = "Clinical-AI-Apollo/Medical-NER" #Named entity recognition

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Define save path
save_path = "saved_medical_ner"

# Save model and tokenizer locally
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f" Model and tokenizer saved successfully in: {save_path}")
