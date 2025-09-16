import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from datetime import datetime
import argparse

class IsskinModelEvaluator:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        
        # Paths to the dataset (relative to this script)
        self.script_dir = Path(__file__).parent
        self.dataset_dir = self.script_dir.parent / "test_dataset"
        self.csv_path = self.dataset_dir / "data.csv"
        
        # Mapping for binary model (malignant = 1, benign = 0)
        self.binary_mapping = {
            'malignant': 1,
            'benign': 0
        }
        
        # Check if API is responding
        self._check_api_health()
        
        # Check if files exist
        self._check_dataset()
    
    def _check_api_health(self):
        try:
            response = requests.get(f"{self.api_url}/ram", timeout=10)
            if response.status_code == 200:
                print(f"API accessible at {self.api_url}")
            else:
                print(f"API responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"\033[91mError connecting to API: {e}\033[0m")
            print("Make sure the backend is running (docker compose up)!")
            sys.exit(1)
    
    def _check_dataset(self):
        if not self.dataset_dir.exists():
            print(f"\033[91mDataset directory not found: {self.dataset_dir}\033[0m")
            print("Make sure the ../test_dataset/ folder exists")
            sys.exit(1)
        
        if not self.csv_path.exists():
            print(f"\033[91mCSV file not found: {self.csv_path}\033[0m")
            print("Make sure the ../test_dataset/data.csv file exists")
            sys.exit(1)

        print(f"Dataset found at: {self.csv_path}")
    
    def load_test_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.csv_path)
            print(f"📊 Loaded {len(df)} records from test dataset")
            
            # Check if required columns exist
            required_cols = ['image', 'binary', 'dx']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in CSV: {missing_cols}")
            
            return df
        except Exception as e:
            print(f"\033[91mError loading CSV: {e}\033[0m")
            sys.exit(1)
    
    def predict_image(self, image_name: str) -> Optional[Dict]:
        image_path = self.dataset_dir / image_name
        
        if not image_path.exists():
            print(f"\033[91mImage not found: {image_path}\033[0m")
            return None
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_name, f, 'image/jpeg')}
                response = requests.post(
                    f"{self.api_url}/predict/",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"\033[91mPrediction error for {image_name}: {response.status_code}\033[0m")
                return None
                
        except Exception as e:
            print(f"\033[91mError processing {image_name}: {e}\033[0m")
            return None
    
    def evaluate_binary_model(self, df: pd.DataFrame) -> Dict:
        # Convert ground truth to numeric values
        y_true = df['binary'].map(self.binary_mapping)
        
        # Extract predictions from binary model
        y_pred = []
        y_pred_proba = []
        
        for _, row in df.iterrows():
            if pd.isna(row['prediction']):
                # If prediction failed, assume majority class
                y_pred.append(0)  # assume benign
                y_pred_proba.append(0.5)
            else:
                pred_data = row['prediction']
                binary_pred = pred_data.get('binary_prediction', {})
                
                # Determine predicted class (highest probability)
                if 'malignant' in binary_pred and 'benign' in binary_pred:
                    malignant_prob = binary_pred['malignant'] / 100.0
                    benign_prob = binary_pred['benign'] / 100.0
                    
                    pred_class = 1 if malignant_prob > benign_prob else 0
                    pred_prob = malignant_prob if pred_class == 1 else benign_prob
                    
                    y_pred.append(pred_class)
                    y_pred_proba.append(pred_prob)
                else:
                    y_pred.append(0)
                    y_pred_proba.append(0.5)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        # Sensitivity (recall for malignant class) = TP / (TP + FN)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred, 
                                                         target_names=['benign', 'malignant'],
                                                         output_dict=True)
        }
    
    def evaluate_dx_model(self, df: pd.DataFrame) -> Dict:
        y_true = df['dx'].tolist()
        y_pred = []
        
        for _, row in df.iterrows():
            if pd.isna(row['prediction']):
                y_pred.append('unknown')
            else:
                pred_data = row['prediction']
                dx_pred = pred_data.get('dx_prediction', {})
                
                # Class with highest probability
                if dx_pred:
                    predicted_class = max(dx_pred.keys(), key=dx_pred.get)
                    y_pred.append(predicted_class)
                else:
                    y_pred.append('unknown')
        
        # Filter cases where we couldn't make predictions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != 'unknown']
        y_true_filtered = [y_true[i] for i in valid_indices]
        y_pred_filtered = [y_pred[i] for i in valid_indices]
        
        if len(y_true_filtered) == 0:
            return {'error': 'No valid predictions found'}
        
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        
        return {
            'accuracy': accuracy,
            'total_samples': len(y_true),
            'valid_predictions': len(y_true_filtered),
            'classification_report': classification_report(y_true_filtered, y_pred_filtered,
                                                         output_dict=True, zero_division=0)
        }
    
    def run_evaluation(self, output_file: Optional[str] = None) -> Dict:
        print("🚀 Starting model evaluation...")
        
        # Load data
        df = self.load_test_data()
        
        # Make predictions for all images
        predictions = []
        total_images = len(df)
        
        for idx, row in df.iterrows():
            print(f"📸 Processing image {idx + 1}/{total_images}: {row['image']}")
            
            prediction = self.predict_image(row['image'])
            predictions.append(prediction)
        
        # Add predictions to DataFrame
        df['prediction'] = predictions
        
        # Evaluate binary model
        print("\n📊 Evaluating binary model...")
        binary_results = self.evaluate_binary_model(df)
        
        # Evaluate dx model
        print("📊 Evaluating dx model...")
        dx_results = self.evaluate_dx_model(df)
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_images': len(df),
                'successful_predictions': len([p for p in predictions if p is not None]),
                'failed_predictions': len([p for p in predictions if p is None])
            },
            'binary_model': binary_results,
            'dx_model': dx_results
        }
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {output_path}")
        
        return results
    
    def print_results(self, results: Dict):
        print("\n" + "="*60)
        print("📈 EVALUATION RESULTS")
        print("="*60)
        
        # Dataset info
        info = results['dataset_info']
        print(f"\n📊 Dataset:")
        print(f"  • Total images: {info['total_images']}")
        print(f"  • Successful predictions: {info['successful_predictions']}")
        print(f"  • Failed predictions: {info['failed_predictions']}")
        
        # Binary Model
        binary = results['binary_model']
        print(f"\n🔍 Binary Model (Malignant vs Benign):")
        print(f"  • Accuracy: {binary['accuracy']:.3f} ({binary['accuracy']*100:.1f}%)")
        print(f"  • Sensitivity: {binary['sensitivity']:.3f} ({binary['sensitivity']*100:.1f}%)")
        print(f"  • Specificity: {binary['specificity']:.3f} ({binary['specificity']*100:.1f}%)")
        
        # DX Model
        dx = results['dx_model']
        if 'error' not in dx:
            print(f"\n🎯 DX Model (Specific Diagnosis):")
            print(f"  • Accuracy: {dx['accuracy']:.3f} ({dx['accuracy']*100:.1f}%)")
            print(f"  • Valid samples: {dx['valid_predictions']}/{dx['total_samples']}")
        else:
            print(f"\n❌ DX Model: {dx['error']}")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Isskin models')
    parser.add_argument('--api-url', default='http://localhost:8000', 
                       help='API URL (default: http://localhost:8000)')
    parser.add_argument('--output', help='Output file to save results (JSON)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = IsskinModelEvaluator(api_url=args.api_url)
    
    # Run evaluation
    results = evaluator.run_evaluation(args.output)
    
    # Show results
    evaluator.print_results(results)


if __name__ == "__main__":
    main()