import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        df = pd.read_csv(file_path)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        # Create a copy of the dataframe
        df_processed = df.copy()
        
        # Handle categorical columns
        categorical_columns = ['protocol', 'attack_type', 'country_src', 'country_dst']
        for col in categorical_columns:
            if col in df_processed.columns:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Drop non-feature columns
        columns_to_drop = ['flow_id', 'honeypot_flag', 'src_ip', 'dst_ip']
        df_processed = df_processed.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Separate features and target
        X = df_processed.drop('is_malicious', axis=1)
        y = df_processed['is_malicious']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_scaler(self, path='models/scaler.joblib'):
        """Save the scaler for later use"""
        joblib.dump(self.scaler, path)
    
    def load_scaler(self, path='models/scaler.joblib'):
        """Load the saved scaler"""
        self.scaler = joblib.load(path)
    
    def transform_single_input(self, input_data):
        """Transform a single input for prediction"""
        return self.scaler.transform(input_data) 