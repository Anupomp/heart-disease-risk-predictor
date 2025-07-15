"""
Heart Disease Prediction Web Interface
Simple web UI for the heart disease prediction system.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import requests
import json

app = Flask(__name__)
app.secret_key = 'heart_disease_predictor_secret_key'

# Configuration
API_URL = 'http://localhost:5000'

@app.route('/')
def index():
    """Main prediction interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    try:
        # Get form data
        patient_data = {
            'age': int(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': int(request.form['trestbps']),
            'chol': int(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': int(request.form['thalach']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }
        
        # Make API request
        response = requests.post(f'{API_URL}/predict', json=patient_data)
        
        if response.status_code == 200:
            result = response.json()
            return render_template('result.html', 
                                 patient_data=patient_data, 
                                 result=result)
        else:
            flash('Error making prediction. Please try again.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/health')
def health():
    """Check API health."""
    try:
        response = requests.get(f'{API_URL}/health')
        return jsonify(response.json())
    except:
        return jsonify({'status': 'API unavailable'}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
