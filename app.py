from flask import Flask, jsonify, render_template, url_for, request, redirect, abort, flash, session
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from models import db, Users
import json
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv # type: ignore
from datetime import datetime
import bcrypt # type: ignore
import os
import pickle
import numpy as np
import fitz
import google.generativeai as gi
from werkzeug.utils import secure_filename

# Load the model
model = pickle.load(open("ml_model.pkl", "rb"))

load_dotenv()

#Get environment variables
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
API_KEY = os.getenv("GOOGLE_API_KEY")

gi.configure(api_key=API_KEY)
if API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set! Check your .env file.")

ADMIN_PASSWORD_HASH = bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///underwriting.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = "uploads"

db.init_app(app)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#Risk Score Calculation
def calculate_risk_score(predicted_claim, age, smoker, bmi):
    min_usd = 823
    max_usd = 58702

    base_risk_score = ((predicted_claim - min_usd) / (max_usd - min_usd)) * 100

    smoker_penalty = 2 if smoker == 1 else 1.0
    high_bmi_penalty = 1.4 if bmi > 30 else 1.0
    if age >= 60:
        high_age_penalty = 3.0
    elif age >= 50 and age < 60:
        high_age_penalty = 2.0
    else:
        high_age_penalty = 1.0
    final_risk_score = base_risk_score * smoker_penalty * high_bmi_penalty * high_age_penalty

    return int(round(min(max(final_risk_score, 0), 100), 0))

#Premium Calculation in INR
def convert_usd_to_inr(predicted_usd, age, smoker, bmi):
    min_premium_inr = 2000
    max_premium_inr = 12000

    min_usd = 823
    max_usd = 58702

    scaled_premium = min_premium_inr + ((predicted_usd - min_usd) / (max_usd - min_usd)) * (max_premium_inr - min_premium_inr)
    
    smoker_penalty = 1.5 if smoker == 1 else 1.0
    high_bmi_penalty = 1.4 if bmi > 30 else 1.0
    if age > 60:
        high_age_penalty = 3.0
    elif age > 50:
        high_age_penalty = 1.5
    else:
        high_age_penalty = 1.0
    final_premium = scaled_premium * smoker_penalty * high_bmi_penalty * high_age_penalty
    
    return int(round(min(max(final_premium, min_premium_inr), max_premium_inr), 2))

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def extract_medical_entities(raw_results):

    save_path = "saved_medical_ner"
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForTokenClassification.from_pretrained(save_path)

    ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = ner_pipeline(raw_results)
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

    for key in structured_data:
        structured_data[key] = list(structured_data[key])

    return structured_data

def get_gemini_response(medical_data):

    prompt = """Generates an underwriter-friendly medical summary using Gemini from a JSON object."""

    prompt = f"""
    Given the following extracted medical data from a patient's discharge summary:

    {medical_data}

    Please generate a structured medical summary for an **insurance underwriter** assessing health risk.
    
    ### **Instructions:**
    - **Ensure proper line breaks between sections.**  
    - **Use plain text with clear headings.**  
    - **Avoid Markdown formatting (`*`, `-`).**  
    - **Maintain clarity and readability.**  

    **Expected Format (STRICTLY FOLLOW THIS):**

    Patient Medical Summary  

    Diagnosis:  
    [Summarize main conditions]  

    Symptoms:  
    [List symptoms]  

    Procedures:  
    [List procedures in clear sentences]  

    Medications:  
    [List medications and dosages in a structured format]  

    Underwriting Considerations:  
    [Key risk insights]  

    Final Risk Classification:  
    [LOW, MEDIUM, or HIGH RISK]  

    **Make sure each section starts on a new line for readability.**
    """

    # Call Gemini API
    model = gi.GenerativeModel('models/gemini-1.5-pro-latest')
    response = model.generate_content([prompt])

    return response.text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/')
def index():
    if session.get('admin'):
        return redirect(url_for('admin_dashboard'))
    if session.get('user'):
        user_id = session.get('user')
        user = Users.query.get_or_404(user_id)
        return redirect(url_for('user', id=user.id, name=user.name))    
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Redirect to admin dashboard if it's the admin
        if email == ADMIN_EMAIL and bcrypt.checkpw(password.encode('utf-8'), ADMIN_PASSWORD_HASH.encode('utf-8')):
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))

        # Check if it's a normal user
        user = Users.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            session['user'] = user.id
            return redirect(url_for('user', id=user.id, name=user.name))
        
        flash('Invalid email or password! Please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        #Get from data
        name = request.form['full_name']
        email = request.form['email']
        password = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Check if email already exists in the database
        if Users.query.filter_by(email=email).first() or email == ADMIN_EMAIL:
            flash('Email already exists! Please log in or use a different email.', 'danger')
            return redirect(url_for('register'))
        # Create new user
        new_user = Users(
            name=name,
            email=email,
            password=password
        )
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Error: Unable to register user. Please try again.', 'danger')
    return render_template('register.html')

@app.route('/user/<int:id>/predict', methods=['POST'])
def user_predict(id):
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    
    sex = 1 if sex.lower() == 'male' else 0
    smoker = 1 if smoker.lower() == 'smoker' else 0
    
    features = np.array([[age, sex, bmi, children, smoker]])
    
    predicted_usd = model.predict(features)[0]

    predicted_inr = convert_usd_to_inr(predicted_usd, age, smoker, bmi)
    risk_score = calculate_risk_score(predicted_usd, age, smoker, bmi)

    return render_template('predict.html', user_id=id, prediction_text=f"Predicted Premium: ₹ {predicted_inr}",risk_score_text=f"Risk Score: {risk_score}")

@app.route('/user/<int:id>/predict')
def user(id):
    if not session.get('user') == id:
        abort(403)
    user = Users.query.get_or_404(id)
    fname = user.name.split()[0]
    return render_template('predict.html', user=user,fname=fname)

@app.route('/analyze_medical_document', methods=['GET', 'POST'])
def analyze_medical_document():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['medical_document']
        if not file or not file.filename.endswith('.pdf'):
            flash("Please upload a valid PDF file!", "danger")
            return redirect(request.url)

        # Save file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_path)

        # Process using NLP
        structured_data = extract_medical_entities(extracted_text)

        # Get summary from Gemini
        summary_text = get_gemini_response(structured_data)

        # Render the results
        return render_template('medical_analysis.html', summary=summary_text, premium=request.form['premium'], risk_score=request.form['risk_score'])

    # Display upload form
    return render_template('upload_medical.html', premium=request.args.get('premium'), risk_score=request.args.get('risk_score'))


if __name__ == '__main__':
    app.run(debug=True)