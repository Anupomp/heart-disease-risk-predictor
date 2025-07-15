import sqlite3
import pandas as pd
import os
from typing import Optional, List, Dict, Any

class DatabaseManager:
    """SQLite database manager for heart disease data with proper schema design"""
    
    def __init__(self, db_path: str = 'heart_disease.db'):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            return self.connection
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def create_database(self, csv_file_path: str):
        """Create database with proper schema and load data"""
        
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file {csv_file_path} not found")
            return False
        
        conn = self.connect()
        if not conn:
            return False
        
        try:
            # Create heart_disease table with proper schema
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS heart_disease (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER NOT NULL CHECK (age > 0 AND age < 120),
                sex INTEGER NOT NULL CHECK (sex IN (0, 1)),
                cp INTEGER NOT NULL CHECK (cp >= 0 AND cp <= 3),
                trestbps INTEGER CHECK (trestbps > 0 AND trestbps < 300),
                chol INTEGER CHECK (chol >= 0),
                fbs INTEGER CHECK (fbs IN (0, 1)),
                restecg INTEGER CHECK (restecg >= 0 AND restecg <= 2),
                thalch INTEGER CHECK (thalch > 0 AND thalch < 250),
                exang TEXT CHECK (exang IN ('TRUE', 'FALSE')),
                oldpeak REAL CHECK (oldpeak >= 0),
                slope INTEGER CHECK (slope >= 0 AND slope <= 2),
                ca INTEGER CHECK (ca >= 0 AND ca <= 4),
                thal INTEGER CHECK (thal >= 0 AND thal <= 3),
                num INTEGER NOT NULL CHECK (num >= 0 AND num <= 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            conn.execute(create_table_sql)
            
            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_age ON heart_disease(age);",
                "CREATE INDEX IF NOT EXISTS idx_sex ON heart_disease(sex);",
                "CREATE INDEX IF NOT EXISTS idx_num ON heart_disease(num);",
                "CREATE INDEX IF NOT EXISTS idx_age_sex ON heart_disease(age, sex);",
                "CREATE INDEX IF NOT EXISTS idx_chol ON heart_disease(chol);",
                "CREATE INDEX IF NOT EXISTS idx_trestbps ON heart_disease(trestbps);"
            ]
            
            for index_sql in indexes:
                conn.execute(index_sql)
            
            # Load CSV data
            df = pd.read_csv(csv_file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Insert data into database
            df.to_sql('heart_disease', conn, if_exists='replace', index=False)
            
            conn.commit()
            print(f"✓ Database created successfully with {len(df)} records")
            print(f"✓ Database file: {self.db_path}")
            
            # Verify data integrity
            self._verify_data_integrity()
            
            return True
            
        except Exception as e:
            print(f"Error creating database: {e}")
            conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        conn = self.connect()
        if not conn:
            return pd.DataFrame()
        
        try:
            if params:
                result = pd.read_sql_query(query, conn, params=params)
            else:
                result = pd.read_sql_query(query, conn)
            return result
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()
        finally:
            self.disconnect()
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get comprehensive table information"""
        conn = self.connect()
        if not conn:
            return {}
        
        try:
            # Table schema
            schema_query = "PRAGMA table_info(heart_disease);"
            schema = pd.read_sql_query(schema_query, conn)
            
            # Table statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT age) as unique_ages,
                MIN(age) as min_age,
                MAX(age) as max_age,
                AVG(age) as avg_age,
                COUNT(DISTINCT sex) as unique_genders
            FROM heart_disease;
            """
            stats = pd.read_sql_query(stats_query, conn)
            
            # Index information
            indexes_query = "PRAGMA index_list(heart_disease);"
            indexes = pd.read_sql_query(indexes_query, conn)
            
            return {
                'schema': schema,
                'statistics': stats,
                'indexes': indexes
            }
            
        except Exception as e:
            print(f"Error getting table info: {e}")
            return {}
        finally:
            self.disconnect()
    
    def _verify_data_integrity(self):
        """Verify data integrity and constraints"""
        conn = self.connect()
        if not conn:
            return
        
        try:
            # Check for constraint violations
            integrity_checks = [
                ("Invalid ages", "SELECT COUNT(*) FROM heart_disease WHERE age <= 0 OR age >= 120"),
                ("Invalid sex values", "SELECT COUNT(*) FROM heart_disease WHERE sex NOT IN (0, 1)"),
                ("Invalid target values", "SELECT COUNT(*) FROM heart_disease WHERE num < 0 OR num > 4"),
                ("Missing critical data", "SELECT COUNT(*) FROM heart_disease WHERE age IS NULL OR sex IS NULL OR num IS NULL")
            ]
            
            print("\n=== DATA INTEGRITY VERIFICATION ===")
            for check_name, query in integrity_checks:
                result = conn.execute(query).fetchone()[0]
                status = "✓ PASS" if result == 0 else f"⚠ FAIL ({result} violations)"
                print(f"{check_name}: {status}")
                
        except Exception as e:
            print(f"Integrity check error: {e}")
        finally:
            self.disconnect()

if __name__ == "__main__":
    # Example usage
    db_manager = DatabaseManager()
    
    # Create database from CSV
    csv_file = 'heart_disease_data.csv'
    if db_manager.create_database(csv_file):
        print("\nDatabase setup completed successfully!")
        
        # Display table information
        info = db_manager.get_table_info()
        if info:
            print("\n=== DATABASE SCHEMA ===")
            print(info['schema'])
            print("\n=== DATABASE STATISTICS ===")
            print(info['statistics'])
    else:
        print("Database setup failed!")
