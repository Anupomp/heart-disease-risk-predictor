import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("=== DATA PREPROCESSING ===")
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Handle boolean columns
        bool_mapping = {'TRUE': 1, 'FALSE': 0, True: 1, False: 0}
        if 'fbs' in df_processed.columns:
            df_processed['fbs'] = df_processed['fbs'].map(bool_mapping)
        if 'exang' in df_processed.columns:
            df_processed['exang'] = df_processed['exang'].map(bool_mapping)
        
        # Handle gender
        if 'sex' in df_processed.columns:
            df_processed['sex'] = df_processed['sex'].map({'Male': 1, 'Female': 0})
        
        # Encode categorical variables
        categorical_columns = ['cp', 'restecg', 'slope', 'thal']
        for col in categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Create binary target variable
        df_processed['target'] = (df_processed['num'] > 0).astype(int)
        
        # Remove invalid data points
        df_processed = df_processed[df_processed['chol'] > 0]  # Remove zero cholesterol
        df_processed = df_processed[df_processed['trestbps'] > 0]  # Remove zero blood pressure
        
        # Feature engineering
        df_processed['age_group'] = pd.cut(df_processed['age'], bins=[0, 45, 55, 65, 100], labels=[0, 1, 2, 3])
        df_processed['chol_risk'] = (df_processed['chol'] > 240).astype(int)
        df_processed['bp_risk'] = (df_processed['trestbps'] > 140).astype(int)
        df_processed['age_chol_interaction'] = df_processed['age'] * df_processed['chol']
        
        # Select features for modeling
        feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                          'age_group', 'chol_risk', 'bp_risk', 'age_chol_interaction']
        
        # Ensure all features exist
        available_features = [col for col in feature_columns if col in df_processed.columns]
        
        self.X = df_processed[available_features]
        self.y = df_processed['target']
        
        print(f"✓ Processed {len(self.X)} samples with {len(available_features)} features")
        print(f"✓ Target distribution: {self.y.value_counts().to_dict()}")
        
        return self
    
    def split_and_scale_data(self, test_size=0.2, random_state=42):
        """Split and scale the data"""
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        
        return self
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n=== MODEL TRAINING ===")
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for models that benefit from it
            if name in ['Logistic Regression', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"✓ {name} - Accuracy: {self.results[name]['accuracy']:.3f}, "
                  f"F1: {self.results[name]['f1']:.3f}, "
                  f"ROC-AUC: {self.results[name]['roc_auc']:.3f}")
        
        return self
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for best models"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        self.tuned_models = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"\nTuning {model_name}...")
            
            base_model = self.models[model_name]
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store best model
            self.tuned_models[model_name] = grid_search.best_estimator_
            
            # Make predictions with tuned model
            y_pred = grid_search.predict(self.X_test)
            y_pred_proba = grid_search.predict_proba(self.X_test)[:, 1]
            
            # Update results
            self.results[f'{model_name} (Tuned)'] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'best_params': grid_search.best_params_
            }
            
            print(f"✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Improved F1 score: {self.results[f'{model_name} (Tuned)']['f1']:.3f}")
        
        return self
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        
        print("\nModel Performance Summary:")
        print(results_df.round(3))
        
        # Find best model
        best_model_name = results_df['f1'].idxmax()
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"F1 Score: {results_df.loc[best_model_name, 'f1']:.3f}")
        
        return results_df
    
    def create_evaluation_plots(self):
        """Create comprehensive evaluation visualizations"""
        print("\n=== CREATING EVALUATION PLOTS ===")
        
        # 1. Model comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Metrics comparison
        results_df = pd.DataFrame(self.results).T
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            results_df[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.capitalize())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # ROC Curves
        ax = axes[1, 2]
        for model_name, result in self.results.items():
            if 'y_pred_proba' in result:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                ax.plot(fpr, tpr, label=f"{model_name} (AUC: {result['roc_auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Confusion matrices for top 3 models
        top_models = pd.DataFrame(self.results).T['f1'].nlargest(3).index
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, model_name in enumerate(top_models):
            cm = confusion_matrix(self.y_test, self.results[model_name]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Get feature importance from Random Forest
        rf_model = self.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features (Random Forest):")
        print(feature_importance.head(10))
        
        # Visualization
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'][::-1])
        plt.yticks(range(len(top_features)), top_features['feature'][::-1])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importance (Random Forest)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_features['importance'][::-1]):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*60)
        print("HEART DISEASE PREDICTION - FINAL REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   • Total samples: {len(self.X)}")
        print(f"   • Features: {len(self.X.columns)}")
        print(f"   • Positive cases: {self.y.sum()} ({self.y.mean()*100:.1f}%)")
        print(f"   • Negative cases: {len(self.y) - self.y.sum()} ({(1-self.y.mean())*100:.1f}%)")
        
        print(f"\n🎯 BEST MODEL PERFORMANCE:")
        results_df = pd.DataFrame(self.results).T
        best_model = results_df['f1'].idxmax()
        best_results = results_df.loc[best_model]
        
        print(f"   • Model: {best_model}")
        print(f"   • Accuracy: {best_results['accuracy']:.3f}")
        print(f"   • Precision: {best_results['precision']:.3f}")
        print(f"   • Recall: {best_results['recall']:.3f}")
        print(f"   • F1-Score: {best_results['f1']:.3f}")
        print(f"   • ROC-AUC: {best_results['roc_auc']:.3f}")
        
        print(f"\n🔍 KEY INSIGHTS:")
        feature_importance = self.feature_importance_analysis()
        top_3_features = feature_importance.head(3)['feature'].tolist()
        print(f"   • Most predictive features: {', '.join(top_3_features)}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • Focus on monitoring: {', '.join(top_3_features[:2])}")
        print(f"   • Model is suitable for: Early screening and risk assessment")
        print(f"   • Recommended for: Healthcare decision support systems")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"   • model_evaluation.png - Model comparison charts")
        print(f"   • confusion_matrices.png - Confusion matrices")
        print(f"   • feature_importance.png - Feature importance chart")
        
        return best_model, best_results

def main():
    """Main execution function"""
    print("🫀 HEART DISEASE RISK PREDICTION SYSTEM")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor('heart_disease_uci.csv')
    
    # Execute full pipeline
    (predictor
     .preprocess_data()
     .split_and_scale_data()
     .train_models()
     .hyperparameter_tuning()
     .evaluate_models()
     .create_evaluation_plots()
     .generate_final_report())
    
    print("\n✅ Heart Disease Prediction Analysis Complete!")
    print("📈 Check generated PNG files for visualizations")

if __name__ == "__main__":
    main()
