import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve, classification_report, f1_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseMLPipeline:
    """
    Enhanced machine learning pipeline with multiple algorithms and comprehensive evaluation.
    Designed for academic Data Management for Data Science course requirements.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.results = {}
    
    def load_data(self, file_path="heart_disease_data.csv"):
        """Load and prepare the heart disease dataset."""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with shape: {self.data.shape}")
            print("Columns:", self.data.columns.tolist())
            return self.data
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None

    
    def prepare_features(self):
        """Prepare features and target variables with proper encoding."""
        if self.data is None:
            print("Error: No data loaded. Please load data first.")
            return None, None
            
        # Convert target to binary (0: no disease, 1: disease)
        # In this dataset, 'num' is the target where 0 = no disease, >0 = disease
        self.data['target'] = (self.data['num'] > 0).astype(int)
        
        # Features and target - drop id, num, and any non-numeric categorical columns
        features_to_drop = ['id', 'num', 'target', 'dataset']  # Remove dataset as it's just a label
        X = self.data.drop(features_to_drop, axis=1)
        y = self.data['target']
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col == 'sex':
                X[col] = (X[col] == 'Male').astype(int)
            elif col in ['cp', 'restecg', 'slope', 'thal']:
                # For other categorical columns, use label encoding or one-hot encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            elif col in ['fbs', 'exang']:
                X[col] = (X[col] == 'TRUE').astype(int)
        
        # Convert all remaining object columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any NaN values with median
        X = X.fillna(X.median())
        
        print(f"Target distribution: No Disease: {(y==0).sum()}, Disease: {(y==1).sum()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Feature names: {X.columns.tolist()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self):
        """Train multiple machine learning models with hyperparameter tuning."""
        if self.X_train is None:
            print("Error: Features not prepared. Please prepare features first.")
            return
        
        # Model configurations
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.05],
                    'max_depth': [3, 5]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
        
        print("Training multiple models with hyperparameter tuning...")
        for name, config in model_configs.items():
            print(f"\nTraining {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            self.models[name] = grid_search.best_estimator_
            
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def evaluate_models(self):
        """Comprehensive evaluation of all trained models."""
        if not self.models:
            print("Error: No models trained. Please train models first.")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)
        
        best_auc = 0
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='roc_auc')
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Track best model
            if auc > best_auc:
                best_auc = auc
                self.best_model = model
                self.best_model_name = name
            
            # Print results
            print(f"\n{name} Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {auc:.4f}")
            print(f"  CV Score:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        print(f"\n🏆 Best Model: {self.best_model_name} (AUC: {best_auc:.4f})")
        
        # Save best model
        joblib.dump(self.best_model, 'best_heart_disease_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        print("✅ Best model and scaler saved!")
    
    def create_visualizations(self):
        """Create comprehensive visualization suite."""
        if not self.results:
            print("Error: No evaluation results. Please evaluate models first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Comparison Bar Plot
        plt.subplot(2, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        model_names = list(self.results.keys())
        
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, model in enumerate(model_names):
            values = [self.results[model][metric] for metric in metrics]
            plt.bar(x + i*width, values, width, label=model, alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*1, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        plt.subplot(2, 3, 2)
        for name in self.results.keys():
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_pred_proba'])
            auc_score = self.results[name]['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Best Model Confusion Matrix
        plt.subplot(2, 3, 3)
        best_predictions = self.results[self.best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 4. Feature Importance (for tree-based models)
        plt.subplot(2, 3, 4)
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature names after preprocessing
            features_to_drop = ['id', 'num', 'target', 'dataset']
            temp_X = self.data.drop(features_to_drop, axis=1)
            categorical_columns = temp_X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in ['sex', 'fbs', 'exang', 'cp', 'restecg', 'slope', 'thal']:
                    pass  # These will be processed
            feature_names = temp_X.columns
            
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.bar(range(len(indices)), importances[indices])
            plt.title(f'Top 10 Feature Importances - {self.best_model_name}')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.ylabel('Importance')
        else:
            plt.text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                    ha='center', va='center', fontsize=12)
            plt.title('Feature Importance')
        
        # 5. Cross-Validation Scores
        plt.subplot(2, 3, 5)
        cv_means = [self.results[model]['cv_mean'] for model in model_names]
        cv_stds = [self.results[model]['cv_std'] for model in model_names]
        
        plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        plt.title('Cross-Validation Scores (Mean ± Std)')
        plt.ylabel('ROC-AUC Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Prediction Distribution
        plt.subplot(2, 3, 6)
        best_probabilities = self.results[self.best_model_name]['y_pred_proba']
        plt.hist(best_probabilities[self.y_test == 0], bins=20, alpha=0.7, 
                label='No Disease', density=True)
        plt.hist(best_probabilities[self.y_test == 1], bins=20, alpha=0.7, 
                label='Disease', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Prediction Probability Distribution - {self.best_model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive visualizations saved as 'comprehensive_model_evaluation.png'")
    
    def run_complete_pipeline(self, file_path="heart_disease_data.csv"):
        """Execute the complete machine learning pipeline."""
        print("🚀 Starting Enhanced Heart Disease ML Pipeline...")
        print("="*60)
        
        # Load data
        if self.load_data(file_path) is None:
            return
        
        # Prepare features
        if self.prepare_features() is None:
            return
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n🎉 Enhanced ML Pipeline completed successfully!")
        print("✅ All models trained and evaluated")
        print("✅ Best model saved for deployment")
        print("✅ Comprehensive visualizations created")


def main():
    """Main execution function."""
    pipeline = HeartDiseaseMLPipeline()
    pipeline.run_complete_pipeline()
    
    # Print summary
    if pipeline.results:
        print(f"\n📊 FINAL SUMMARY:")
        print(f"Best Model: {pipeline.best_model_name}")
        print(f"Best AUC Score: {pipeline.results[pipeline.best_model_name]['auc']:.4f}")
        print(f"Models Evaluated: {len(pipeline.results)}")
        print("Ready for clinical deployment! 🏥")


if __name__ == "__main__":
    main()
