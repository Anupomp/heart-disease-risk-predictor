"""
Comprehensive Exploratory Data Analysis and Visualization Module
Enhanced with features from comparative analysis to create the most robust implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEDA:
    """
    Advanced Exploratory Data Analysis with interactive visualizations
    and statistical insights for heart disease prediction.
    """
    
    def __init__(self, data_path="heart_disease_data.csv"):
        """Initialize the EDA class with data loading and preprocessing."""
        self.data = pd.read_csv(data_path)
        self.processed_data = None
        self.feature_names = None
        self.target_name = 'num'
        self.setup_data()
        
    def setup_data(self):
        """Prepare data for analysis."""
        # Create binary target
        self.data['target'] = (self.data['num'] > 0).astype(int)
        
        # Identify feature columns
        self.feature_names = [col for col in self.data.columns 
                             if col not in ['id', 'num', 'target', 'dataset']]
        
        print(f"Dataset loaded with {len(self.data)} samples and {len(self.feature_names)} features")
        print(f"Target distribution: {self.data['target'].value_counts().to_dict()}")
    
    def create_basic_statistics_dashboard(self):
        """Create comprehensive basic statistics dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Heart Disease Dataset - Basic Statistics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Age distribution
        axes[0,0].hist(self.data['age'], bins=25, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0,0].axvline(self.data['age'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.data["age"].mean():.1f}')
        axes[0,0].set_title('Age Distribution')
        axes[0,0].set_xlabel('Age (years)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Target distribution
        target_counts = self.data['target'].value_counts()
        colors = ['lightcoral', 'lightblue']
        wedges, texts, autotexts = axes[0,1].pie(target_counts.values, 
                                                labels=['No Disease', 'Disease'], 
                                                colors=colors, autopct='%1.1f%%', 
                                                startangle=90)
        axes[0,1].set_title('Disease Distribution')
        
        # 3. Cholesterol distribution by disease status
        no_disease = self.data[self.data['target'] == 0]['chol']
        disease = self.data[self.data['target'] == 1]['chol']
        
        axes[0,2].hist(no_disease, bins=20, alpha=0.7, label='No Disease', color='green')
        axes[0,2].hist(disease, bins=20, alpha=0.7, label='Disease', color='red')
        axes[0,2].set_title('Cholesterol Distribution by Disease Status')
        axes[0,2].set_xlabel('Cholesterol Level')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Gender analysis
        gender_analysis = self.data.groupby('sex').agg({
            'target': 'mean',
            'age': 'mean',
            'chol': 'mean',
            'trestbps': 'mean'
        }).round(2)
        
        x_pos = np.arange(len(gender_analysis.index))
        axes[1,0].bar(x_pos, gender_analysis['target'], color=['pink', 'lightblue'], alpha=0.8)
        axes[1,0].set_title('Disease Rate by Gender')
        axes[1,0].set_xlabel('Gender (0=Female, 1=Male)')
        axes[1,0].set_ylabel('Disease Rate')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(['Female', 'Male'])
        for i, v in enumerate(gender_analysis['target']):
            axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Chest pain type analysis
        cp_analysis = self.data.groupby('cp').agg({
            'target': 'mean',
            'cp': 'count'
        }).rename(columns={'cp': 'count'})
        
        axes[1,1].bar(cp_analysis.index, cp_analysis['target'], 
                     color='orange', alpha=0.8)
        axes[1,1].set_title('Disease Rate by Chest Pain Type')
        axes[1,1].set_xlabel('Chest Pain Type')
        axes[1,1].set_ylabel('Disease Rate')
        for i, v in enumerate(cp_analysis['target']):
            axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Blood pressure vs heart rate scatter
        scatter = axes[1,2].scatter(self.data['trestbps'], self.data['thalch'], 
                                   c=self.data['target'], cmap='viridis', 
                                   alpha=0.7, s=50)
        axes[1,2].set_title('Blood Pressure vs Max Heart Rate')
        axes[1,2].set_xlabel('Resting Blood Pressure')
        axes[1,2].set_ylabel('Max Heart Rate')
        plt.colorbar(scatter, ax=axes[1,2], label='Disease (0=No, 1=Yes)')
        
        plt.tight_layout()
        plt.savefig('basic_statistics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Basic Statistics Dashboard created and saved!")
    
    def create_correlation_analysis(self):
        """Create comprehensive correlation analysis."""
        # Prepare numeric data
        numeric_data = self.data[self.feature_names + ['target']].select_dtypes(include=[np.number])
        
        # Create correlation matrix
        correlation_matrix = numeric_data.corr()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Comprehensive Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Full correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, ax=axes[0,0], 
                   cmap='coolwarm', center=0, annot=True, fmt='.2f',
                   square=True, cbar_kws={"shrink": .8})
        axes[0,0].set_title('Feature Correlation Matrix')
        
        # 2. Target correlations
        target_corr = correlation_matrix['target'].drop('target').sort_values(key=abs, ascending=False)
        colors = ['red' if x < 0 else 'green' for x in target_corr.values]
        axes[0,1].barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        axes[0,1].set_yticks(range(len(target_corr)))
        axes[0,1].set_yticklabels(target_corr.index)
        axes[0,1].set_title('Feature Correlation with Heart Disease')
        axes[0,1].set_xlabel('Correlation Coefficient')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Scatter plot matrix for top correlated features
        top_features = target_corr.head(4).index.tolist()
        top_data = numeric_data[top_features + ['target']]
        
        # Create scatter plot matrix
        pd.plotting.scatter_matrix(top_data, ax=axes[1,:], figsize=(12, 8), 
                                 alpha=0.7, diagonal='hist', c=self.data['target'], 
                                 cmap='viridis')
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print insights
        print("🔍 CORRELATION INSIGHTS:")
        print(f"Strongest positive correlation with disease: {target_corr.iloc[0]:.3f} ({target_corr.index[0]})")
        print(f"Strongest negative correlation with disease: {target_corr.iloc[-1]:.3f} ({target_corr.index[-1]})")
        
        return correlation_matrix
    
    def create_advanced_visualizations(self):
        """Create advanced statistical visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('Advanced Statistical Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Box plots for key features
        key_features = ['age', 'chol', 'trestbps', 'thalch']
        for i, feature in enumerate(key_features):
            row, col = i // 2, i % 2
            sns.boxplot(data=self.data, x='target', y=feature, ax=axes[0, col] if row == 0 else axes[1, col])
            axes[0, col if row == 0 else 1, col].set_title(f'{feature.title()} by Disease Status')
            axes[0, col if row == 0 else 1, col].set_xlabel('Disease (0=No, 1=Yes)')
        
        # 2. Violin plots
        sns.violinplot(data=self.data, x='target', y='oldpeak', ax=axes[0,2])
        axes[0,2].set_title('ST Depression Distribution by Disease')
        
        # 3. Age group analysis
        self.data['age_group'] = pd.cut(self.data['age'], 
                                       bins=[0, 40, 50, 60, 100], 
                                       labels=['<40', '40-50', '50-60', '60+'])
        age_group_stats = self.data.groupby('age_group')['target'].agg(['count', 'mean'])
        
        axes[1,2].bar(age_group_stats.index, age_group_stats['mean'], 
                     color='skyblue', alpha=0.8)
        axes[1,2].set_title('Disease Rate by Age Group')
        axes[1,2].set_ylabel('Disease Rate')
        for i, v in enumerate(age_group_stats['mean']):
            axes[1,2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 4. Exercise angina analysis
        exercise_stats = self.data.groupby('exang')['target'].agg(['count', 'mean'])
        axes[2,0].bar(['No Exercise Angina', 'Exercise Angina'], exercise_stats['mean'], 
                     color=['green', 'red'], alpha=0.8)
        axes[2,0].set_title('Disease Rate by Exercise-Induced Angina')
        axes[2,0].set_ylabel('Disease Rate')
        for i, v in enumerate(exercise_stats['mean']):
            axes[2,0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 5. Cholesterol categories
        self.data['chol_category'] = pd.cut(self.data['chol'], 
                                           bins=[0, 200, 240, 1000], 
                                           labels=['Normal', 'Borderline', 'High'])
        chol_stats = self.data.groupby('chol_category')['target'].agg(['count', 'mean'])
        
        axes[2,1].bar(chol_stats.index, chol_stats['mean'], 
                     color=['green', 'yellow', 'red'], alpha=0.8)
        axes[2,1].set_title('Disease Rate by Cholesterol Category')
        axes[2,1].set_ylabel('Disease Rate')
        for i, v in enumerate(chol_stats['mean']):
            axes[2,1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 6. Multiple risk factors analysis
        def calculate_risk_score(row):
            score = 0
            if row['age'] > 55: score += 1
            if row['chol'] > 240: score += 1
            if row['trestbps'] > 140: score += 1
            if row['exang'] == 1: score += 1
            return score
        
        self.data['risk_score'] = self.data.apply(calculate_risk_score, axis=1)
        risk_stats = self.data.groupby('risk_score')['target'].agg(['count', 'mean'])
        
        axes[2,2].bar(risk_stats.index, risk_stats['mean'], 
                     color=plt.cm.Reds(np.linspace(0.3, 1, len(risk_stats))), alpha=0.8)
        axes[2,2].set_title('Disease Rate by Risk Factor Count')
        axes[2,2].set_xlabel('Number of Risk Factors')
        axes[2,2].set_ylabel('Disease Rate')
        for i, v in enumerate(risk_stats['mean']):
            axes[2,2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('advanced_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Advanced Visualizations created and saved!")
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Age vs Max Heart Rate', 'Cholesterol Distribution',
                           'Chest Pain Type Analysis', 'Blood Pressure by Gender',
                           'Feature Correlation Heatmap', 'Risk Factor Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None]]
        )
        
        # 1. Age vs Max Heart Rate scatter
        fig.add_trace(
            go.Scatter(
                x=self.data[self.data['target']==0]['age'],
                y=self.data[self.data['target']==0]['thalch'],
                mode='markers',
                name='No Disease',
                marker=dict(color='green', size=6, opacity=0.7)
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data[self.data['target']==1]['age'],
                y=self.data[self.data['target']==1]['thalch'],
                mode='markers',
                name='Disease',
                marker=dict(color='red', size=6, opacity=0.7)
            ), row=1, col=1
        )
        
        # 2. Cholesterol histogram
        fig.add_trace(
            go.Histogram(
                x=self.data[self.data['target']==0]['chol'],
                name='No Disease',
                opacity=0.7,
                marker_color='green'
            ), row=1, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=self.data[self.data['target']==1]['chol'],
                name='Disease',
                opacity=0.7,
                marker_color='red'
            ), row=1, col=2
        )
        
        # 3. Chest pain analysis
        cp_stats = self.data.groupby('cp')['target'].mean()
        fig.add_trace(
            go.Bar(
                x=cp_stats.index,
                y=cp_stats.values,
                name='Disease Rate by CP Type',
                marker_color='orange'
            ), row=2, col=1
        )
        
        # 4. Blood pressure by gender
        bp_gender = self.data.groupby(['sex', 'target'])['trestbps'].mean().unstack()
        fig.add_trace(
            go.Bar(
                x=['Female', 'Male'],
                y=bp_gender[0],
                name='No Disease BP',
                marker_color='lightblue'
            ), row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=['Female', 'Male'],
                y=bp_gender[1],
                name='Disease BP',
                marker_color='lightcoral'
            ), row=2, col=2
        )
        
        # 5. Correlation heatmap
        numeric_data = self.data[self.feature_names].select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ), row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Interactive Heart Disease Analysis Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save interactive dashboard
        fig.write_html("interactive_dashboard.html")
        fig.show()
        
        print("✅ Interactive Dashboard created and saved as 'interactive_dashboard.html'!")
    
    def create_dimensionality_reduction_analysis(self):
        """Create PCA and t-SNE visualizations."""
        # Prepare data for dimensionality reduction
        numeric_features = self.data[self.feature_names].select_dtypes(include=[np.number])
        
        # Handle missing values and scale data
        numeric_features = numeric_features.fillna(numeric_features.median())
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        
        # PCA Analysis
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        # t-SNE Analysis
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(scaled_features)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dimensionality Reduction Analysis', fontsize=16, fontweight='bold')
        
        # PCA scatter plot
        scatter1 = axes[0,0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                    c=self.data['target'], cmap='viridis', alpha=0.7)
        axes[0,0].set_title(f'PCA Analysis\n(Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter1, ax=axes[0,0], label='Disease')
        
        # t-SNE scatter plot
        scatter2 = axes[0,1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                    c=self.data['target'], cmap='viridis', alpha=0.7)
        axes[0,1].set_title('t-SNE Analysis')
        axes[0,1].set_xlabel('t-SNE 1')
        axes[0,1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[0,1], label='Disease')
        
        # PCA component analysis
        feature_names_numeric = numeric_features.columns
        pc1_loadings = pca.components_[0]
        pc2_loadings = pca.components_[1]
        
        axes[1,0].barh(range(len(feature_names_numeric)), pc1_loadings, alpha=0.7)
        axes[1,0].set_yticks(range(len(feature_names_numeric)))
        axes[1,0].set_yticklabels(feature_names_numeric, fontsize=10)
        axes[1,0].set_title('PC1 Feature Loadings')
        axes[1,0].set_xlabel('Loading Value')
        
        axes[1,1].barh(range(len(feature_names_numeric)), pc2_loadings, alpha=0.7, color='orange')
        axes[1,1].set_yticks(range(len(feature_names_numeric)))
        axes[1,1].set_yticklabels(feature_names_numeric, fontsize=10)
        axes[1,1].set_title('PC2 Feature Loadings')
        axes[1,1].set_xlabel('Loading Value')
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Dimensionality Reduction Analysis created and saved!")
        
        return pca, tsne, pca_result, tsne_result
    
    def create_statistical_tests_summary(self):
        """Create comprehensive statistical tests summary."""
        print("=" * 80)
        print("COMPREHENSIVE STATISTICAL TESTS SUMMARY")
        print("=" * 80)
        
        results = {}
        
        # 1. T-tests for continuous variables
        continuous_vars = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        
        print("\n1. T-TESTS FOR CONTINUOUS VARIABLES:")
        print("-" * 50)
        
        for var in continuous_vars:
            group0 = self.data[self.data['target'] == 0][var]
            group1 = self.data[self.data['target'] == 1][var]
            
            # Remove NaN values
            group0 = group0.dropna()
            group1 = group1.dropna()
            
            t_stat, p_value = stats.ttest_ind(group0, group1)
            
            print(f"{var.upper()}:")
            print(f"  No Disease: {group0.mean():.2f} ± {group0.std():.2f}")
            print(f"  Disease:    {group1.mean():.2f} ± {group1.std():.2f}")
            print(f"  T-statistic: {t_stat:.3f}, P-value: {p_value:.6f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            print()
            
            results[f'{var}_ttest'] = {'t_stat': t_stat, 'p_value': p_value}
        
        # 2. Chi-square tests for categorical variables
        categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        print("2. CHI-SQUARE TESTS FOR CATEGORICAL VARIABLES:")
        print("-" * 50)
        
        for var in categorical_vars:
            if var in self.data.columns:
                contingency_table = pd.crosstab(self.data[var], self.data['target'])
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                print(f"{var.upper()}:")
                print(f"  Chi-square: {chi2_stat:.3f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Degrees of freedom: {dof}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                print()
                
                results[f'{var}_chi2'] = {'chi2_stat': chi2_stat, 'p_value': p_value, 'dof': dof}
        
        # 3. Correlation analysis
        print("3. CORRELATION ANALYSIS:")
        print("-" * 50)
        
        numeric_data = self.data[self.feature_names + ['target']].select_dtypes(include=[np.number])
        correlations = numeric_data.corr()['target'].drop('target')
        
        print("Feature correlations with heart disease:")
        for feature, corr in correlations.sort_values(key=abs, ascending=False).items():
            print(f"  {feature}: {corr:.4f}")
        
        return results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive EDA report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
        print("=" * 80)
        
        # Dataset overview
        print(f"\nDATASET OVERVIEW:")
        print(f"• Total samples: {len(self.data)}")
        print(f"• Total features: {len(self.feature_names)}")
        print(f"• Missing values: {self.data.isnull().sum().sum()}")
        print(f"• Target distribution: {dict(self.data['target'].value_counts())}")
        
        # Run all analyses
        print("\nGenerating visualizations...")
        self.create_basic_statistics_dashboard()
        correlation_matrix = self.create_correlation_analysis()
        self.create_advanced_visualizations()
        self.create_interactive_dashboard()
        pca, tsne, pca_result, tsne_result = self.create_dimensionality_reduction_analysis()
        statistical_results = self.create_statistical_tests_summary()
        
        print("\n" + "=" * 80)
        print("EDA REPORT GENERATION COMPLETED!")
        print("=" * 80)
        print("\nGenerated files:")
        print("• basic_statistics_dashboard.png")
        print("• correlation_analysis.png")
        print("• advanced_visualizations.png")
        print("• interactive_dashboard.html")
        print("• dimensionality_reduction_analysis.png")
        
        return {
            'correlation_matrix': correlation_matrix,
            'pca': pca,
            'tsne': tsne,
            'statistical_results': statistical_results
        }

def main():
    """Main execution function."""
    print("🚀 Starting Comprehensive Exploratory Data Analysis...")
    
    # Initialize EDA
    eda = ComprehensiveEDA()
    
    # Generate comprehensive report
    results = eda.generate_comprehensive_report()
    
    print("\n✅ Comprehensive EDA completed successfully!")
    print("🎨 All visualizations and interactive dashboard created!")
    print("📊 Statistical analysis completed!")
    
    return results

if __name__ == "__main__":
    main()
