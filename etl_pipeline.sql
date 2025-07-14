-- Create table
CREATE TABLE heart_data (
    id SERIAL PRIMARY KEY,
    age INT CHECK (age BETWEEN 20 AND 100),
    sex VARCHAR(1) CHECK (sex IN ('M', 'F')),
    cholesterol INT CHECK (cholesterol > 0),
    rest_bp INT CHECK (rest_bp > 0),
    cp_type VARCHAR(10),
    fbs BOOLEAN,
    max_hr INT CHECK (max_hr > 0),
    ex_angina BOOLEAN,
    target BOOLEAN
);

-- Import CSV (PostgreSQL \COPY command from psql)
-- \COPY heart_data(age, sex, cholesterol, rest_bp, cp_type, fbs, max_hr, ex_angina, target)
-- FROM 'heart.csv' DELIMITER ',' CSV HEADER;

-- Indexing for performance
CREATE INDEX idx_age ON heart_data(age);
CREATE INDEX idx_cholesterol ON heart_data(cholesterol);
CREATE INDEX idx_target ON heart_data(target);
