"""
Heart Disease Prediction API
Production-ready prediction service for heart disease risk assessment.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Union
import logging
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

class HeartDiseasePredictorAPI:
    """Production API for heart disease prediction."""
    
    def __init__(self, model_path: str = "best_heart_disease_model.pkl", scaler_path: str = "feature_scaler.pkl"):
        """Initialize the prediction API."""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.logger = logging.getLogger(__name__)
            
            # Feature names (must match training data)
            self.feature_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal'
            ]
            
            self.logger.info("Heart Disease Predictor API initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API: {e}")
            raise
    
    def validate_input(self, data: Dict) -> Dict:
        """Validate input data."""
        errors = []
        
        # Check required features
        for feature in self.feature_names:
            if feature not in data:
                errors.append(f"Missing required feature: {feature}")
        
        # Validate ranges
        validations = {
            'age': (0, 120),
            'sex': (0, 1),
            'cp': (0, 3),
            'trestbps': (50, 300),
            'chol': (0, 600),
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (50, 250),
            'oldpeak': (0, 10),
            'slope': (0, 2),
            'ca': (0, 4),
            'thal': (0, 3)
        }
        
        for feature, (min_val, max_val) in validations.items():
            if feature in data:
                if not (min_val <= data[feature] <= max_val):
                    errors.append(f"{feature} value {data[feature]} outside valid range [{min_val}, {max_val}]")
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def predict(self, patient_data: Dict) -> Dict:
        """Make prediction for a single patient."""
        try:
            # Validate input
            validation = self.validate_input(patient_data)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'Invalid input data',
                    'details': validation['errors']
                }
            
            # Prepare features
            features = np.array([[patient_data[feature] for feature in self.feature_names]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Interpret results
            risk_level = 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
            
            return {
                'success': True,
                'prediction': int(prediction),
                'probability_no_disease': float(probability[0]),
                'probability_disease': float(probability[1]),
                'risk_level': risk_level,
                'confidence': float(max(probability))
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_predict(self, patients_data: List[Dict]) -> List[Dict]:
        """Make predictions for multiple patients."""
        results = []
        
        for i, patient_data in enumerate(patients_data):
            try:
                result = self.predict(patient_data)
                result['patient_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'patient_id': i,
                    'success': False,
                    'error': str(e)
                })
        
        return results

# Initialize the predictor
try:
    predictor = HeartDiseasePredictorAPI()
except Exception as e:
    print(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Heart Disease Predictor API',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict_single():
    """Single patient prediction endpoint."""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = predictor.predict(data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'patients' not in data:
            return jsonify({'error': 'No patients data provided'}), 400
        
        results = predictor.batch_predict(data['patients'])
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/example', methods=['GET'])
def get_example():
    """Get example input format."""
    example = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    return jsonify({
        'example_input': example,
        'feature_descriptions': {
            'age': 'Age in years (0-120)',
            'sex': 'Gender (0=female, 1=male)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (50-300)',
            'chol': 'Serum cholesterol (0-600)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (0,1)',
            'restecg': 'Resting ECG results (0-2)',
            'thalach': 'Maximum heart rate achieved (50-250)',
            'oldpeak': 'ST depression induced by exercise (0-10)',
            'slope': 'Slope of peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels (0-4)',
            'thal': 'Thalassemia type (0-3)'
        }
    })

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
