from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import json
import fitz
import google.generativeai as gi
import os
from dotenv import load_dotenv # type: ignore

# Load model and tokenizer from local path
save_path = "saved_medical_ner"
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForTokenClassification.from_pretrained(save_path)

# Initialize NLP Pipeline
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# Example usage
pdf_path = "./sample texts/sample1.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

raw_results = ner_pipeline(pdf_text)

# Print extracted medical terms
for entity in raw_results:
    print(f"{entity['entity_group']}: {entity['word']} (Score: {entity['score']:.2f})")

def format_entities(results):
    structured_data = {
        "Diseases": set(),
        "Medications": set(),
        "Procedures": set(),
        "Symptoms": set(),
        "Dosages": set(),
        "Lab Tests": set(),
        "Therapeutic Procedures": set()
    }

    # Mapping entity labels to structured categories
    for entity in results:
        category = entity["entity_group"]
        text = entity["word"]

        if "DISEASE" in category or "DISORDER" in category:
            structured_data["Diseases"].add(text)
        elif "MEDICATION" in category:
            structured_data["Medications"].add(text)
        elif "PROCEDURE" in category or "THERAPEUTIC_PROCEDURE" in category:
            structured_data["Procedures"].add(text)
        elif "SIGN_SYMPTOM" in category:
            structured_data["Symptoms"].add(text)
        elif "DOSAGE" in category:
            structured_data["Dosages"].add(text)
        elif "DIAGNOSTIC_PROCEDURE" in category:
            structured_data["Lab Tests"].add(text)
        elif "THERAPEUTIC_PROCEDURE" in category:
            structured_data["Therapeutic Procedures"].add(text)

    # Convert sets back to lists for JSON serialization
    for key in structured_data:
        structured_data[key] = list(structured_data[key])

    return structured_data

# Format the extracted results
structured_output = format_entities(raw_results)

# Convert to JSON and print
json_output = json.dumps(structured_output, indent=4)
print(json_output)

######   Gemini API Response   ##########


load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure API
gi.configure(api_key=API_KEY)

if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set! Check your .env file.")

def get_gemini_response(medical_data):
    """Generates a human-readable medical summary using Gemini from a JSON object."""

    # Convert JSON object to a formatted string
    # medical_data_str = json.dumps(medical_data, indent=4)

    # Define the prompt for Gemini
    prompt = f"""
    Given the following medical data extracted from a patient's discharge summary:

    {medical_data}

    Please generate a professional, structured, and concise medical summary (IN A SINGLE PARAGRAPH) that explains the 
    patient's conditions, symptoms, procedures, medications, and dosages in a human-readable format.
    Ensure the summary is easy to understand for both medical professionals and patients.
    AT THE END, GIVE IN ONE LINE EXPLANATION THE PATIENT'S RISK FROM THE LIST ['LOW RISK','MEDIUM RISK','HIGH RISK'] SUCH THAT IT HELPS THE MEDICAL UNDERWRITER.
    """

    # Call Gemini API
    model = gi.GenerativeModel('models/gemini-1.5-pro-latest')
    response = model.generate_content([prompt])

    return response.text

summary_text = get_gemini_response(json_output)
print(summary_text)