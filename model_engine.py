import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap
import joblib
import os

class ModelEngine:
    def __init__(self):
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        self.explainer = None
        self.feature_names = None
        self.model_path = 'model_artifacts.joblib'

    def train(self, data_input='data/processed_features.csv'):
        if isinstance(data_input, pd.DataFrame):
            df = data_input
        else:
            print(f"Loading data from {data_input}...")
            if not os.path.exists(data_input):
                raise FileNotFoundError(f"{data_input} not found. Please run data_gen.py first.")
            df = pd.read_csv(data_input)
        
        # Handle user_id if it's a column (it is, from to_csv default)
        if 'user_id' in df.columns:
            df = df.set_index('user_id')
            
        # Prepare features
        # Drop non-feature columns
        # We drop 'dept' because we rely on z-scores which are dept-normalized
        X = df.drop(columns=['dept', 'is_insider'], errors='ignore')
        y = df['is_insider']
        self.feature_names = X.columns.tolist()
        
        print(f"Training on {len(X)} records with features: {self.feature_names}")

        # Layer 1: Isolation Forest
        print("Training Layer 1: Isolation Forest...")
        self.iso_forest.fit(X)
        # decision_function returns negative for anomalies, positive for normal.
        # We want an anomaly score where higher is more anomalous.
        # So we take negative of decision_function.
        anomaly_scores = -self.iso_forest.decision_function(X)
        
        # Add anomaly score as a feature for Layer 2
        X_layer2 = X.copy()
        X_layer2['anomaly_score'] = anomaly_scores

        # Layer 2: XGBoost
        print("Training Layer 2: XGBoost...")
        # Use stratify to ensure insiders are in both train and test
        X_train, X_test, y_train, y_test = train_test_split(X_layer2, y, test_size=0.2, random_state=42, stratify=y)
        
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.xgb_model.predict(X_test)
        y_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))
        
        try:
            print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        except ValueError:
            print("ROC AUC: Undefined (only one class in test set)")
            
        # Feature Importance
        print("\nFeature Importance:")
        importances = self.xgb_model.feature_importances_
        feature_imp = pd.DataFrame({'Feature': X_layer2.columns, 'Importance': importances})
        feature_imp = feature_imp.sort_values('Importance', ascending=False)
        print(feature_imp)
        
        # Train SHAP explainer
        print("Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.xgb_model)
        
        print("Training complete.")
        return X_layer2, y

    def predict(self, X):
        """
        Predicts risk probability for new data.
        X: DataFrame with original features (no anomaly_score)
        """
        # 1. Get Anomaly Score
        anomaly_scores = -self.iso_forest.decision_function(X)
        
        # 2. Prepare for XGBoost
        X_layer2 = X.copy()
        X_layer2['anomaly_score'] = anomaly_scores
        
        # 3. Predict Probability
        probs = self.xgb_model.predict_proba(X_layer2)[:, 1]
        
        return probs, X_layer2

    def get_reason_codes(self, X_layer2_row):
        """
        Returns top 3 reason codes for a single user (row).
        X_layer2_row: Series or DataFrame row with all features including anomaly_score
        """
        # Ensure input is 2D
        if isinstance(X_layer2_row, pd.Series):
            X_layer2_row = X_layer2_row.to_frame().T
            
        shap_values = self.explainer.shap_values(X_layer2_row)
        
        # For binary classification, shap_values might be a list or array
        # XGBoost usually returns array (n_samples, n_features)
        if isinstance(shap_values, list):
             sv = shap_values[1][0]
        else:
            sv = shap_values[0]
            
        # Get indices of top 3 contributors (absolute value? or just positive?)
        # Usually we care about what pushes the risk UP.
        indices = np.argsort(sv)[::-1]
        top_indices = indices[:3]
        
        reasons = []
        feature_names = X_layer2_row.columns.tolist()
        
        for idx in top_indices:
            reasons.append({
                'feature': feature_names[idx],
                'value': float(X_layer2_row.iloc[0, idx]),
                'shap_value': float(sv[idx])
            })
            
        return reasons

    def save(self):
        os.makedirs('models', exist_ok=True)
        self.model_path = 'models/threat_detector.pkl'
        joblib.dump({
            'iso_forest': self.iso_forest,
            'xgb_model': self.xgb_model,
            'explainer': self.explainer,
            'feature_names': self.feature_names
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        self.model_path = 'models/threat_detector.pkl'
        if not os.path.exists(self.model_path):
            # Fallback for backward compatibility if needed, or just fail
            if os.path.exists('model_artifacts.joblib'):
                 self.model_path = 'model_artifacts.joblib'
            else:
                raise FileNotFoundError(f"Model file {self.model_path} not found. Train first.")
            
        artifacts = joblib.load(self.model_path)
        self.iso_forest = artifacts['iso_forest']
        self.xgb_model = artifacts['xgb_model']
        self.explainer = artifacts['explainer']
        self.feature_names = artifacts['feature_names']
        print("Model loaded.")

    @staticmethod
    def process_logs(df):
        """Calculates features for the model from raw logs."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # 1. After Hours Ratio
        # Define after hours as < 6 AM or > 6 PM (18:00)
        df['is_after_hours'] = (df['hour'] < 6) | (df['hour'] > 18)
        
        user_stats = df.groupby('user_id').agg({
            'is_after_hours': 'mean', # Ratio of after hours events
            'volume_mb': 'sum',
            'activity_type': 'count',
            'is_insider': 'max' # If user was ever flagged as insider (or just use the truth label)
        }).rename(columns={
            'is_after_hours': 'after_hours_ratio',
            'volume_mb': 'total_volume_mb',
            'activity_type': 'total_activities'
        })
        
        # 2. Deviation from Peer Group (Dept)
        # We need to join back to get dept for each user
        if 'dept' in df.columns:
            user_depts = df[['user_id', 'dept']].drop_duplicates().set_index('user_id')
            user_stats = user_stats.join(user_depts)
            
            # Calculate dept averages
            dept_stats = user_stats.groupby('dept').agg({
                'total_volume_mb': ['mean', 'std'],
                'total_activities': ['mean', 'std']
            })
            
            # Flatten columns
            dept_stats.columns = ['_'.join(col) for col in dept_stats.columns]
            
            # Join dept stats to user stats
            user_stats = user_stats.join(dept_stats, on='dept')
            
            # Calculate Z-scores
            user_stats['volume_z_score'] = (user_stats['total_volume_mb'] - user_stats['total_volume_mb_mean']) / (user_stats['total_volume_mb_std'] + 1e-5)
            user_stats['activity_z_score'] = (user_stats['total_activities'] - user_stats['total_activities_mean']) / (user_stats['total_activities_std'] + 1e-5)
        else:
            # Fallback if no dept info
            user_stats['dept'] = 'Unknown'
            user_stats['volume_z_score'] = 0
            user_stats['activity_z_score'] = 0
        
        # Clean up
        features_df = user_stats[['after_hours_ratio', 'total_volume_mb', 'total_activities', 'volume_z_score', 'activity_z_score', 'dept', 'is_insider']]
        
        return features_df

if __name__ == "__main__":
    engine = ModelEngine()
    engine.train()
    engine.save()
