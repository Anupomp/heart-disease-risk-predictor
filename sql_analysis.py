import pandas as pd
import sqlite3
from database_setup import DatabaseManager

class SQLAnalyzer:
    def __init__(self, db_path='heart_disease.db'):
        self.db_manager = DatabaseManager(db_path)
    
    def exploratory_analysis(self):
        """Perform comprehensive SQL-based exploratory data analysis"""
        
        print("=== SQL EXPLORATORY DATA ANALYSIS ===")
        
        # 1. Basic Statistics
        print("\n1. BASIC DATASET STATISTICS")
        basic_stats = self.db_manager.execute_query("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT age) as unique_ages,
                COUNT(DISTINCT sex) as unique_genders,
                MIN(age) as min_age,
                MAX(age) as max_age,
                AVG(age) as avg_age,
                MIN(chol) as min_cholesterol,
                MAX(chol) as max_cholesterol,
                AVG(chol) as avg_cholesterol
            FROM heart_disease
        """)
        print(basic_stats)
        
        # 2. Target Variable Distribution
        print("\n2. HEART DISEASE DISTRIBUTION")
        target_dist = self.db_manager.execute_query("""
            SELECT 
                num as heart_disease_level,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM heart_disease), 2) as percentage
            FROM heart_disease 
            GROUP BY num 
            ORDER BY num
        """)
        print(target_dist)
        
        # 3. Age Group Analysis
        print("\n3. AGE GROUP ANALYSIS")
        age_analysis = self.db_manager.execute_query("""
            SELECT 
                CASE 
                    WHEN age < 40 THEN 'Under 40'
                    WHEN age BETWEEN 40 AND 50 THEN '40-50'
                    WHEN age BETWEEN 51 AND 60 THEN '51-60'
                    ELSE 'Over 60'
                END as age_group,
                COUNT(*) as total_patients,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent
            FROM heart_disease 
            GROUP BY age_group 
            ORDER BY MIN(age)
        """)
        print(age_analysis)
        
        # 4. Gender-based Analysis
        print("\n4. GENDER-BASED ANALYSIS")
        gender_analysis = self.db_manager.execute_query("""
            SELECT 
                sex,
                COUNT(*) as total_patients,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent,
                ROUND(AVG(age), 1) as avg_age,
                ROUND(AVG(chol), 1) as avg_cholesterol
            FROM heart_disease 
            GROUP BY sex
        """)
        print(gender_analysis)
        
        # 5. Chest Pain Type Analysis
        print("\n5. CHEST PAIN TYPE ANALYSIS")
        cp_analysis = self.db_manager.execute_query("""
            SELECT 
                cp as chest_pain_type,
                COUNT(*) as total_patients,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent
            FROM heart_disease 
            GROUP BY cp 
            ORDER BY disease_rate_percent DESC
        """)
        print(cp_analysis)
        
        # 6. Cholesterol Level Analysis
        print("\n6. CHOLESTEROL LEVEL ANALYSIS")
        chol_analysis = self.db_manager.execute_query("""
            SELECT 
                CASE 
                    WHEN chol < 200 THEN 'Normal (<200)'
                    WHEN chol BETWEEN 200 AND 239 THEN 'Borderline (200-239)'
                    ELSE 'High (>=240)'
                END as cholesterol_level,
                COUNT(*) as total_patients,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent
            FROM heart_disease 
            WHERE chol > 0  -- Exclude invalid cholesterol readings
            GROUP BY cholesterol_level 
            ORDER BY MIN(chol)
        """)
        print(chol_analysis)
        
        # 7. Blood Pressure Analysis
        print("\n7. BLOOD PRESSURE ANALYSIS")
        bp_analysis = self.db_manager.execute_query("""
            SELECT 
                CASE 
                    WHEN trestbps < 120 THEN 'Normal (<120)'
                    WHEN trestbps BETWEEN 120 AND 129 THEN 'Elevated (120-129)'
                    WHEN trestbps BETWEEN 130 AND 139 THEN 'Stage 1 High (130-139)'
                    ELSE 'Stage 2 High (>=140)'
                END as blood_pressure_category,
                COUNT(*) as total_patients,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent
            FROM heart_disease 
            WHERE trestbps > 0  -- Exclude invalid BP readings
            GROUP BY blood_pressure_category 
            ORDER BY MIN(trestbps)
        """)
        print(bp_analysis)
        
        # 8. Exercise Induced Angina Analysis
        print("\n8. EXERCISE INDUCED ANGINA ANALYSIS")
        exang_analysis = self.db_manager.execute_query("""
            SELECT 
                CASE WHEN exang = 'TRUE' THEN 'Has Exercise Angina' ELSE 'No Exercise Angina' END as exercise_angina,
                COUNT(*) as total_patients,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent
            FROM heart_disease 
            GROUP BY exang
        """)
        print(exang_analysis)
        
        # 9. Complex Analysis - High Risk Patients
        print("\n9. HIGH RISK PATIENT IDENTIFICATION")
        high_risk = self.db_manager.execute_query("""
            SELECT 
                COUNT(*) as high_risk_patients,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent
            FROM heart_disease 
            WHERE age > 55 
                AND chol > 240 
                AND trestbps > 140
        """)
        print(high_risk)
        
        # 10. Dataset Quality Check
        print("\n10. DATA QUALITY ASSESSMENT")
        quality_check = self.db_manager.execute_query("""
            SELECT 
                'Age' as field,
                COUNT(*) as total_records,
                COUNT(CASE WHEN age IS NOT NULL AND age > 0 AND age < 120 THEN 1 END) as valid_records,
                COUNT(CASE WHEN age IS NULL OR age <= 0 OR age >= 120 THEN 1 END) as invalid_records
            FROM heart_disease
            
            UNION ALL
            
            SELECT 
                'Cholesterol' as field,
                COUNT(*) as total_records,
                COUNT(CASE WHEN chol IS NOT NULL AND chol > 0 AND chol < 600 THEN 1 END) as valid_records,
                COUNT(CASE WHEN chol IS NULL OR chol <= 0 OR chol >= 600 THEN 1 END) as invalid_records
            FROM heart_disease
            
            UNION ALL
            
            SELECT 
                'Blood Pressure' as field,
                COUNT(*) as total_records,
                COUNT(CASE WHEN trestbps IS NOT NULL AND trestbps > 0 AND trestbps < 250 THEN 1 END) as valid_records,
                COUNT(CASE WHEN trestbps IS NULL OR trestbps <= 0 OR trestbps >= 250 THEN 1 END) as invalid_records
            FROM heart_disease
        """)
        print(quality_check)
    
    def advanced_queries(self):
        """Execute advanced SQL queries for deeper insights"""
        
        print("\n=== ADVANCED SQL ANALYSIS ===")
        
        # 1. Correlation-like analysis using SQL
        print("\n1. FEATURE CORRELATION WITH DISEASE (SQL-based)")
        correlation_analysis = self.db_manager.execute_query("""
            SELECT 
                'Age > 60' as feature,
                COUNT(*) as total,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END), 3) as disease_rate
            FROM heart_disease 
            WHERE age > 60
            
            UNION ALL
            
            SELECT 
                'High Cholesterol (>240)' as feature,
                COUNT(*) as total,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END), 3) as disease_rate
            FROM heart_disease 
            WHERE chol > 240
            
            UNION ALL
            
            SELECT 
                'High Blood Pressure (>140)' as feature,
                COUNT(*) as total,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END), 3) as disease_rate
            FROM heart_disease 
            WHERE trestbps > 140
            
            ORDER BY disease_rate DESC
        """)
        print(correlation_analysis)
        
        # 2. Multi-factor risk analysis
        print("\n2. MULTI-FACTOR RISK ANALYSIS")
        multi_factor = self.db_manager.execute_query("""
            SELECT 
                CASE 
                    WHEN age > 55 AND chol > 240 AND trestbps > 140 THEN '3 Risk Factors'
                    WHEN (age > 55 AND chol > 240) OR (age > 55 AND trestbps > 140) OR (chol > 240 AND trestbps > 140) THEN '2 Risk Factors'
                    WHEN age > 55 OR chol > 240 OR trestbps > 140 THEN '1 Risk Factor'
                    ELSE 'No Major Risk Factors'
                END as risk_category,
                COUNT(*) as patient_count,
                SUM(CASE WHEN num > 0 THEN 1 ELSE 0 END) as with_disease,
                ROUND(AVG(CASE WHEN num > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as disease_rate_percent
            FROM heart_disease 
            GROUP BY risk_category
            ORDER BY disease_rate_percent DESC
        """)
        print(multi_factor)

if __name__ == "__main__":
    # First, ensure database exists
    db_manager = DatabaseManager()
    try:
        # Try to read from existing database
        test_query = db_manager.execute_query("SELECT COUNT(*) FROM heart_disease")
        print("Database found, proceeding with analysis...")
    except:
        # Create database if it doesn't exist
        print("Creating database...")
        db_manager.create_database('heart_disease_data.csv')
    
    # Perform SQL analysis
    analyzer = SQLAnalyzer()
    analyzer.exploratory_analysis()
    analyzer.advanced_queries()
