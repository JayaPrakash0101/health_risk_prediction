# Health Insurance Risk Assessment & Medical Document Analysis System

This project is a web-based health insurance underwriting support system that uses a Machine Learning model (XGBoost) to predict an applicant's risk score and insurance premium, and an NLP-based Named Entity Recognition (NER) model to extract insights from uploaded medical documents.

---

## 🚀 Features

- **User Authentication:** Register and login functionality for secure access.
- **Risk & Premium Prediction:** Predicts applicant risk score and premium from structured health data (age, BMI, smoking, etc.).
- **Medical Document Analysis:** Extracts diseases, symptoms, medications, procedures, and generates a medical summary using Gemini API.
- **PDF Upload Support:** Upload discharge summaries in PDF format for analysis.
  
---

## ⚙️ Tech Stack

- **Frontend:** HTML5, CSS, Bootstrap, Jinja2 Templates
- **Backend:** Python (Flask)
- **ML Model:** XGBoost (saved as `ml_model.pkl`)
- **NLP Model:** Clinical-AI-Apollo/Medical-NER (Hugging Face Transformers)
- **Database:** SQLite (underwriting.db)
- **PDF Text Extraction:** PyMuPDF
- **LLM Integration:** Google Gemini API for summarization
- **Others:** bcrypt, dotenv, numpy, pandas, transformers

---

## 📁 Project Structure

```
major_project/
├── app.py                   # Main Flask application
├── predict.py               # ML model training and testing (run this with uncommenting code for pickle file generation once to create .pkl file)
├── models.py                # SQLAlchemy models
├── db_init.py               # DB initialization
├── ml_model.pkl             # Trained XGBoost model (will be generated when uncommented predict.py file is run)
├── instance/
│   └── underwriting.db      # SQLite database
├── static/
│   ├── css/
│   └── images/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── predict.html
│   ├── upload_medical.html
│   └── medical_analysis.html
├── saved_medical_ner/       # Folder which will be created by running save_nlp_model.py
├── sample texts/
│   └── sample1.pdf          # Sample discharge summary
├── .env                     # Environment variables
├── save_nlp_model.py        # Script to download and save NER model
└── requirements.txt         # Python dependencies
```

---

## 🧪 Setup Instructions

### 🔧 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/health-underwriting-app.git
cd health-underwriting-app
```

### 🔒 Step 2: Create and Activate Virtual Environment
```bash
python -m venv env
env\Scripts\activate     # On Windows
```

### 📦 Step 3: Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 🔑 Step 4: Configure Environment Variables

Create a `.env` file in the root directory and add:
```
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=your_admin_password
GOOGLE_API_KEY=your_gemini_api_key
```

### 📥 Step 5: Download and Save the NLP Model

Run the following script to automatically download and store the pretrained NLP model:

```bash
python save_nlp_model.py
```

> ⚠️ This step is **mandatory** to ensure medical document processing works correctly.

---
### 🗄️ Step 6: Initialize the Database

Run the following script once to create the SQLite database and necessary tables:

```bash
python db_init.py
```
### 📊 Step 7: Train and Save the ML Model

To generate the `ml_model.pkl` file used for premium and risk score prediction, run the following:

```bash
python predict.py
```

### ▶️ Run the Application

```bash
python app.py
```

Then open your browser and visit:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📚 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co)
- [Medical Insurance Dataset](https://www.kaggle.com/datasets/rajgupta2019/medical-insurance-dataset)
- [Google Gemini API](https://ai.google.dev)

---