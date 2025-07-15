import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseVisualizer:
    def __init__(self, df):
        self.df = df.copy()
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_comprehensive_eda(self):
        """Create comprehensive exploratory data analysis visualizations"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Target variable distribution
        plt.subplot(4, 3, 1)
        target_counts = self.df['num'].value_counts().sort_index()
        plt.pie(target_counts.values, labels=[f'Level {i}' for i in target_counts.index], autopct='%1.1f%%')
        plt.title('Heart Disease Severity Distribution', fontsize=14, fontweight='bold')
        
        # 2. Age distribution
        plt.subplot(4, 3, 2)
        plt.hist(self.df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.df['age'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["age"].mean():.1f}')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution')
        plt.legend()
        
        # 3. Gender distribution
        plt.subplot(4, 3, 3)
        gender_counts = self.df['sex'].value_counts()
        plt.bar(gender_counts.index, gender_counts.values, color=['pink', 'lightblue'])
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.title('Gender Distribution')
        plt.xticks(rotation=45)
        
        # 4. Chest pain type distribution
        plt.subplot(4, 3, 4)
        cp_counts = self.df['cp'].value_counts()
        plt.barh(range(len(cp_counts)), cp_counts.values)
        plt.yticks(range(len(cp_counts)), cp_counts.index)
        plt.xlabel('Count')
        plt.title('Chest Pain Type Distribution')
        
        # 5. Cholesterol distribution
        plt.subplot(4, 3, 5)
        # Filter out zero cholesterol values for better visualization
        chol_filtered = self.df[self.df['chol'] > 0]['chol']
        plt.hist(chol_filtered, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(chol_filtered.mean(), color='red', linestyle='--', label=f'Mean: {chol_filtered.mean():.1f}')
        plt.xlabel('Cholesterol Level')
        plt.ylabel('Frequency')
        plt.title('Cholesterol Distribution (>0)')
        plt.legend()
        
        # 6. Blood pressure distribution
        plt.subplot(4, 3, 6)
        bp_filtered = self.df[self.df['trestbps'] > 0]['trestbps']
        plt.hist(bp_filtered, bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.axvline(bp_filtered.mean(), color='red', linestyle='--', label=f'Mean: {bp_filtered.mean():.1f}')
        plt.xlabel('Resting Blood Pressure')
        plt.ylabel('Frequency')
        plt.title('Blood Pressure Distribution')
        plt.legend()
        
        # 7. Heart rate distribution
        plt.subplot(4, 3, 7)
        plt.hist(self.df['thalch'], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(self.df['thalch'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["thalch"].mean():.1f}')
        plt.xlabel('Max Heart Rate Achieved')
        plt.ylabel('Frequency')
        plt.title('Maximum Heart Rate Distribution')
        plt.legend()
        
        # 8. Exercise induced angina
        plt.subplot(4, 3, 8)
        exang_counts = self.df['exang'].value_counts()
        plt.pie(exang_counts.values, labels=['No Exercise Angina', 'Exercise Angina'], autopct='%1.1f%%')
        plt.title('Exercise Induced Angina')
        
        # 9. Fasting blood sugar
        plt.subplot(4, 3, 9)
        fbs_counts = self.df['fbs'].value_counts()
        plt.bar(['Normal (<120)', 'High (>120)'], fbs_counts.values, color=['lightblue', 'salmon'])
        plt.ylabel('Count')
        plt.title('Fasting Blood Sugar Distribution')
        
        # 10. ST depression (oldpeak)
        plt.subplot(4, 3, 10)
        plt.hist(self.df['oldpeak'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('ST Depression (oldpeak)')
        plt.ylabel('Frequency')
        plt.title('ST Depression Distribution')
        
        # 11. Number of major vessels
        plt.subplot(4, 3, 11)
        ca_counts = self.df['ca'].value_counts().sort_index()
        plt.bar(ca_counts.index, ca_counts.values, color='gold')
        plt.xlabel('Number of Major Vessels')
        plt.ylabel('Count')
        plt.title('Major Vessels Colored by Fluoroscopy')
        
        # 12. Thalassemia type
        plt.subplot(4, 3, 12)
        thal_counts = self.df['thal'].value_counts()
        plt.barh(range(len(thal_counts)), thal_counts.values)
        plt.yticks(range(len(thal_counts)), thal_counts.index)
        plt.xlabel('Count')
        plt.title('Thalassemia Type Distribution')
        
        plt.tight_layout()
        plt.savefig('comprehensive_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_analysis(self):
        """Create correlation analysis visualizations"""
        
        # Prepare numerical data for correlation
        numerical_df = self.df.select_dtypes(include=[np.number]).copy()
        
        # Convert categorical to numerical for correlation
        df_corr = self.df.copy()
        df_corr['sex_numeric'] = df_corr['sex'].map({'Male': 1, 'Female': 0})
        df_corr['cp_numeric'] = pd.Categorical(df_corr['cp']).codes
        df_corr['restecg_numeric'] = pd.Categorical(df_corr['restecg']).codes
        df_corr['slope_numeric'] = pd.Categorical(df_corr['slope']).codes
        df_corr['thal_numeric'] = pd.Categorical(df_corr['thal']).codes
        df_corr['fbs_numeric'] = df_corr['fbs'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
        df_corr['exang_numeric'] = df_corr['exang'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
        
        # Select numerical columns for correlation
        corr_columns = ['age', 'sex_numeric', 'cp_numeric', 'trestbps', 'chol', 'fbs_numeric', 
                       'restecg_numeric', 'thalch', 'exang_numeric', 'oldpeak', 'slope_numeric', 
                       'ca', 'thal_numeric', 'num']
        
        correlation_matrix = df_corr[corr_columns].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance visualization (correlation with target)
        target_corr = correlation_matrix['num'].abs().sort_values(ascending=False)[1:]  # Exclude self-correlation
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if x > 0.3 else 'orange' if x > 0.2 else 'lightblue' for x in target_corr.values]
        bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors)
        plt.yticks(range(len(target_corr)), target_corr.index)
        plt.xlabel('Absolute Correlation with Heart Disease')
        plt.title('Feature Importance (Correlation with Target)', fontsize=14, fontweight='bold')
        plt.axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='Strong Correlation (>0.3)')
        plt.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Moderate Correlation (>0.2)')
        plt.legend()
        
        # Add value labels on bars
        for i, v in enumerate(target_corr.values):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_disease_analysis(self):
        """Create disease-specific analysis visualizations"""
        
        # Create binary target for easier analysis
        df_analysis = self.df.copy()
        df_analysis['has_disease'] = (df_analysis['num'] > 0).astype(int)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Age vs Disease
        age_groups = pd.cut(df_analysis['age'], bins=[0, 40, 50, 60, 100], labels=['<40', '40-50', '50-60', '60+'])
        age_disease = pd.crosstab(age_groups, df_analysis['has_disease'], normalize='index') * 100
        age_disease.plot(kind='bar', ax=axes[0,0], color=['lightblue', 'salmon'])
        axes[0,0].set_title('Heart Disease Rate by Age Group')
        axes[0,0].set_ylabel('Percentage')
        axes[0,0].legend(['No Disease', 'Has Disease'])
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Gender vs Disease
        gender_disease = pd.crosstab(df_analysis['sex'], df_analysis['has_disease'], normalize='index') * 100
        gender_disease.plot(kind='bar', ax=axes[0,1], color=['lightblue', 'salmon'])
        axes[0,1].set_title('Heart Disease Rate by Gender')
        axes[0,1].set_ylabel('Percentage')
        axes[0,1].legend(['No Disease', 'Has Disease'])
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Chest Pain vs Disease
        cp_disease = pd.crosstab(df_analysis['cp'], df_analysis['has_disease'], normalize='index') * 100
        cp_disease.plot(kind='bar', ax=axes[0,2], color=['lightblue', 'salmon'])
        axes[0,2].set_title('Heart Disease Rate by Chest Pain Type')
        axes[0,2].set_ylabel('Percentage')
        axes[0,2].legend(['No Disease', 'Has Disease'])
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Cholesterol levels vs Disease
        chol_groups = pd.cut(df_analysis[df_analysis['chol'] > 0]['chol'], 
                           bins=[0, 200, 240, 1000], labels=['Normal', 'Borderline', 'High'])
        chol_disease_df = df_analysis[df_analysis['chol'] > 0].copy()
        chol_disease_df['chol_group'] = chol_groups
        chol_disease = pd.crosstab(chol_disease_df['chol_group'], chol_disease_df['has_disease'], normalize='index') * 100
        chol_disease.plot(kind='bar', ax=axes[1,0], color=['lightblue', 'salmon'])
        axes[1,0].set_title('Heart Disease Rate by Cholesterol Level')
        axes[1,0].set_ylabel('Percentage')
        axes[1,0].legend(['No Disease', 'Has Disease'])
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Exercise Angina vs Disease
        exang_disease = pd.crosstab(df_analysis['exang'], df_analysis['has_disease'], normalize='index') * 100
        exang_disease.plot(kind='bar', ax=axes[1,1], color=['lightblue', 'salmon'])
        axes[1,1].set_title('Heart Disease Rate by Exercise Angina')
        axes[1,1].set_ylabel('Percentage')
        axes[1,1].legend(['No Disease', 'Has Disease'])
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Number of vessels vs Disease
        ca_disease = pd.crosstab(df_analysis['ca'], df_analysis['has_disease'], normalize='index') * 100
        ca_disease.plot(kind='bar', ax=axes[1,2], color=['lightblue', 'salmon'])
        axes[1,2].set_title('Heart Disease Rate by Number of Major Vessels')
        axes[1,2].set_ylabel('Percentage')
        axes[1,2].legend(['No Disease', 'Has Disease'])
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('disease_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_statistical_insights(self):
        """Create visualizations showing statistical insights"""
        
        print("=== STATISTICAL INSIGHTS ===")
        
        # Chi-square tests for categorical variables
        categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        target_binary = (self.df['num'] > 0).astype(int)
        
        chi2_results = []
        for var in categorical_vars:
            if var in self.df.columns:
                contingency_table = pd.crosstab(self.df[var], target_binary)
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                chi2_results.append({
                    'Variable': var,
                    'Chi2': chi2,
                    'P-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
        
        chi2_df = pd.DataFrame(chi2_results)
        print("\nChi-square Test Results (Categorical vs Target):")
        print(chi2_df.to_string(index=False))
        
        # Create visualization of statistical significance
        plt.figure(figsize=(12, 6))
        colors = ['red' if sig == 'Yes' else 'lightblue' for sig in chi2_df['Significant']]
        bars = plt.bar(chi2_df['Variable'], -np.log10(chi2_df['P-value']), color=colors)
        plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='Significance threshold (p=0.05)')
        plt.xlabel('Variables')
        plt.ylabel('-log10(p-value)')
        plt.title('Statistical Significance of Categorical Variables')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Add value labels on bars
        for bar, p_val in zip(bars, chi2_df['P-value']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{p_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_eda_report(csv_path):
    """Generate comprehensive EDA report"""
    print("=== HEART DISEASE EDA REPORT ===")
    print("Loading data and generating visualizations...")
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Initialize visualizer
    visualizer = HeartDiseaseVisualizer(df)
    
    # Generate all visualizations
    print("\n1. Creating comprehensive exploratory data analysis...")
    visualizer.create_comprehensive_eda()
    
    print("\n2. Creating correlation analysis...")
    visualizer.create_correlation_analysis()
    
    print("\n3. Creating disease-specific analysis...")
    visualizer.create_disease_analysis()
    
    print("\n4. Creating statistical insights...")
    visualizer.create_statistical_insights()
    
    print("\n✓ All visualizations saved as PNG files")
    print("✓ EDA Report completed successfully!")

if __name__ == "__main__":
    generate_eda_report('heart_disease_uci.csv')
