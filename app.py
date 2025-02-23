from flask import Flask, render_template, jsonify, request
import requests
import os
import json
import easyocr
import torch
import base64
import google.generativeai as genai
from PIL import Image
import numpy as np
import tensorflow as tf

# -------------------------------
# Gemini API and EasyOCR Setup
# -------------------------------

# Configure Gemini API with your API key
genai.configure(api_key="AIzaSyAs4gxFQ8ylLqoYK7AIKmtvBIphbLKPuKc")  # Replace with your actual API key

app = Flask(__name__)

# Initialize EasyOCR reader (for English)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# -------------------------------
# Skin Cancer Detection Setup
# -------------------------------

SCD_CLASSES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Lichenoid Keratosis",
    "Dermatofibroma",
    "Melanocytic Nevus",
    "Pyogenic Granuloma",
    "Melanoma"
]

# Load the pre-trained skin cancer detection model
SCD_MODEL = tf.keras.models.load_model('best_model.h5')

# -------------------------------
# Helper Function: Gemini Extraction
# -------------------------------

def extract_data_with_gemini_text(extracted_text):
    prompt = (
        "You are a helpful medical assistant. Analyze the following medical report text and extract "
        "the following details if available:\n"
        "1. Patient Name\n"
        "2. Age\n"
        "3. Sex\n"
        "4. Hospital Name\n"
        "5. Doctor Name\n"
        "6. Diagnosis\n\n"
        "Return the answer as valid JSON in the following format:\n"
        '{ "Patient Name": "", "Age": "", "Sex": "", "Hospital Name": "", "Doctor Name": "", "Diagnosis": "" }\n\n'
        "Report text:\n" + extracted_text
    )
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    print("DEBUG: Gemini API Output:")
    print(response.text)
    try:
        json_output = json.loads(response.text)
    except json.JSONDecodeError:
        json_output = {"error": "Invalid JSON returned by Gemini", "raw_output": response.text}
    return json_output

# -------------------------------
# Routes
# -------------------------------

# Main landing page
@app.route('/')
def home():
    return render_template("index.html")

# Sign-up page
@app.route('/signup', methods=['GET'])
def signup():
    return render_template("sign-up.html")

# -------------------------------
# 1. Search Functionality
# -------------------------------
@app.route('/search')
def search():
    address = request.args.get("address", "").strip()
    data = None  # Initialize data as None

    if address:
        query = f'Cancer doctors in {address}'
        api_url = f'https://edc-pict.site/api/search-places?query="{query}"'
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            data = {"error": str(e)}
    
    return render_template("namelist.html", data=data)


# -------------------------------
# 2. Medical Report Extraction
# -------------------------------
@app.route('/medical', methods=['GET'])
def medical_form():
    return render_template('medical.html')

@app.route('/medical/upload', methods=['POST'])
def medical_upload():
    if 'report' not in request.files:
        return "No file uploaded.", 400

    file = request.files['report']
    if file.filename == '':
        return "No selected file.", 400

    # Save the uploaded file temporarily.
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    # Use EasyOCR to extract text from the image.
    try:
        results = reader.readtext(file_path, detail=0)
        extracted_text = "\n".join(results)
        print("DEBUG: OCR Extracted Text:")
        print(extracted_text)
    except Exception as e:
        return f"Error reading file with EasyOCR: {e}", 400

    # Process the extracted text with Gemini.
    structured_data = extract_data_with_gemini_text(extracted_text)

    # Clean up the temporary file.
    os.remove(file_path)

    # Return the structured JSON result as pretty-printed HTML.
    return f"<pre>{json.dumps(structured_data, indent=2)}</pre>"

# -------------------------------
# 3. Skin Cancer Detection
# -------------------------------
@app.route('/skin', methods=['GET'])
def skin_form():
    return render_template('skin_home.html')

@app.route('/skin/results', methods=['POST'])
def skin_results():
    age = request.form.get("age")
    sex = request.form.get("sex")
    region = request.form.get("region")
    
    if "pic" not in request.files:
        return "No file uploaded.", 400

    pic = request.files["pic"]

    try:
        inputimg = Image.open(pic)
        inputimg = inputimg.resize((28, 28))
        img = np.array(inputimg).reshape(-1, 28, 28, 3)
    except Exception as e:
        return f"Error processing image: {e}", 400

    # Predict the lesion type using the pre-trained model.
    result = SCD_MODEL.predict(img)
    result = result.tolist()
    max_prob = max(result[0])
    class_ind = result[0].index(max_prob)
    disease = SCD_CLASSES[class_ind]

    # Provide additional information based on the prediction.
    if class_ind == 0:
        info = "Actinic keratosis is a pre-malignant lesion due to intraepithelial keratinocyte dysplasia."
    elif class_ind == 1:
        info = ("Basal cell carcinoma is a type of skin cancer that starts in the basal cells, "
                "commonly seen in sun-exposed areas.")
    elif class_ind == 2:
        info = ("Benign lichenoid keratosis typically presents as a solitary lesion; "
                "its exact pathogenesis remains unclear.")
    elif class_ind == 3:
        info = ("Dermatofibromas are benign skin growths most frequently seen on the lower legs, upper arms, or back.")
    elif class_ind == 4:
        info = ("A melanocytic nevus, or mole, is a melanocytic tumor that contains nevus cells.")
    elif class_ind == 5:
        info = ("Pyogenic granulomas are small, red skin growths that bleed easily due to a high concentration of blood vessels.")
    elif class_ind == 6:
        info = ("Melanoma is the most serious form of skin cancer; it arises from melanocytes and is strongly linked to UV exposure.")
    else:
        info = "Unknown condition."

    # Create a prompt that includes the user’s details and predicted disease.
    prompt = (
        f"User details:\n"
        f"- Age: {age}\n"
        f"- Sex: {sex}\n"
        f"- Lesion Region: {region}\n"
        f"- Predicted Disease: {disease}\n\n"
        f"Based on these details, provide a clear and actionable recommendation on what the user should do next."
    )
    
    # Generate a recommendation using Gemini.
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        recommendation = response.text
    except Exception as e:
        recommendation = f"Error generating recommendation: {str(e)}"
    
    return render_template("results.html", result=disease, info=info, recommendation=recommendation)

# -------------------------------
# Run the App
# -------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
