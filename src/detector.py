import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)

    def detect_statistical(self, df, threshold=3):
        """Detect anomalies using statistical Z-score"""
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if df[col].nunique() < 3:
                continue
                
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                for idx in df.index:
                    value = df.loc[idx, col]
                    deviation = (value - mean) / std
                    
                    if abs(deviation) > threshold:
                        if idx not in anomalies:
                            anomalies[idx] = []
                        anomalies[idx].append({
                            'type': 'statistical',
                            'column': col,
                            'value': float(value),
                            'mean': float(mean),
                            'std': float(std),
                            'deviation': float(deviation),
                            'issue': f'Statistical deviation (Z-score: {deviation:.2f})',
                            'severity': 'High' if abs(deviation) > 5 else 'Medium'
                        })
        return anomalies

    def detect_business_rules(self, df):
        """Detect anomalies based on business rules"""
        anomalies = {}
        
        if 'amount' in df.columns:
            # Negative amounts
            for idx in df[df['amount'] < 0].index:
                if idx not in anomalies:
                    anomalies[idx] = []
                anomalies[idx].append({
                    'type': 'business_rule',
                    'column': 'amount',
                    'value': float(df.loc[idx, 'amount']),
                    'issue': 'Negative value',
                    'severity': 'High'
                })
            
            # Extremely high amounts (>99th percentile)
            if len(df['amount']) > 0:
                threshold_99 = df['amount'].quantile(0.99)
                for idx in df[df['amount'] > threshold_99].index:
                    if idx not in anomalies:
                        anomalies[idx] = []
                    # Check if not already added as statistical outlier
                    already_added = any(
                        issue['column'] == 'amount' and issue['type'] == 'statistical' 
                        for issue in anomalies[idx]
                    )
                    if not already_added:
                        anomalies[idx].append({
                            'type': 'business_rule',
                            'column': 'amount',
                            'value': float(df.loc[idx, 'amount']),
                            'issue': f'Extreme value (>{threshold_99:.2f})',
                            'severity': 'Medium'
                        })
        return anomalies

    def detect_data_quality(self, df):
        """Detect data quality issues"""
        anomalies = {}
        
        if 'category' in df.columns:
            for idx in df.index:
                cat_value = df.loc[idx, 'category']
                if pd.isna(cat_value) or cat_value in ['', None]:
                    if idx not in anomalies:
                        anomalies[idx] = []
                    anomalies[idx].append({
                        'type': 'data_quality',
                        'column': 'category',
                        'value': 'NULL',
                        'issue': 'Missing category',
                        'severity': 'Low'
                    })
                elif cat_value in ['Unknown', 'UNKNOWN', 'Suspicious']:
                    if idx not in anomalies:
                        anomalies[idx] = []
                    anomalies[idx].append({
                        'type': 'data_quality',
                        'column': 'category',
                        'value': str(cat_value),
                        'issue': 'Suspicious category',
                        'severity': 'Medium'
                    })
        return anomalies

    def detect_isolation_forest(self, df):
        """Detect anomalies using Isolation Forest (ML)"""
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return anomalies
            
        # Fill NA for ML
        data_for_ml = df[numeric_cols].fillna(0)
        
        try:
            # Fit and predict
            predictions = self.iso_forest.fit_predict(data_for_ml)
            # -1 is anomaly, 1 is normal
            
            anomaly_indices = np.where(predictions == -1)[0]
            
            for idx in anomaly_indices:
                real_idx = df.index[idx]
                if real_idx not in anomalies:
                    anomalies[real_idx] = []
                
                anomalies[real_idx].append({
                    'type': 'ml_isolation_forest',
                    'column': 'multivariate',
                    'value': 'Complex Pattern',
                    'issue': 'Detected by Isolation Forest Algorithm',
                    'severity': 'Medium'
                })
        except Exception as e:
            print(f"ML Detection failed: {e}")
            
        return anomalies

    def detect_all(self, df, threshold=3, use_ml=True):
        """Run all detection methods"""
        all_anomalies = {}
        
        # Helper to merge anomalies
        def merge_anomalies(new_anomalies):
            for idx, issues in new_anomalies.items():
                if idx not in all_anomalies:
                    all_anomalies[idx] = []
                all_anomalies[idx].extend(issues)

        merge_anomalies(self.detect_statistical(df, threshold))
        merge_anomalies(self.detect_business_rules(df))
        merge_anomalies(self.detect_data_quality(df))
        
        if use_ml:
            merge_anomalies(self.detect_isolation_forest(df))
            
        return all_anomalies
