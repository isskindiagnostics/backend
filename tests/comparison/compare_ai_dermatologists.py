import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import json
from datetime import datetime
import argparse
import traceback

class IsskinvsDermatologistsComparison:
    def __init__(self, api_url: str = "http://localhost:8000"):
        print(f"Initializing comparison with API URL: {api_url}")
        
        self.api_url = api_url.rstrip('/')
        
        # Paths to dataset (relative to this script)
        self.script_dir = Path(__file__).parent
        self.dataset_dir = self.script_dir.parent / "test_dataset"
        self.csv_path = self.dataset_dir / "data.csv"
        
        print(f"Script directory: {self.script_dir}")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"CSV path: {self.csv_path}")
        
        # Model dx categories
        self.dx_categories = [
            "ak", "bcc", "df", "lentigo", "melanoma", "nevus",
            "scc", "seborrheic_keratosis", "vasc"
        ]
        
        # Binary classification mapping
        self.binary_mapping = {
            'malignant': 1,
            'benign': 0
        }
        
        # Clinical impression to dx category mapping
        self.clinical_to_dx_mapping = {
            # Malignant lesions
            'melanoma': 'melanoma',
            'melanoma maligno': 'melanoma',
            'melanoma invasivo': 'melanoma',
            'melanoma in situ': 'melanoma',
            'basal cell carcinoma': 'bcc',
            'carcinoma basocelular': 'bcc',
            'bcc': 'bcc',
            'squamous cell carcinoma': 'scc',
            'carcinoma espinocelular': 'scc',
            'carcinoma de células escamosas': 'scc',
            'scc': 'scc',
            'actinic keratosis': 'ak',
            'queratose actínica': 'ak',
            'queratose actinica': 'ak',
            'ak': 'ak',
            
            # Benign lesions
            'nevus': 'nevus',
            'nevo': 'nevus',
            'nevo melanocítico': 'nevus',
            'nevo melanocitico': 'nevus',
            'melanocytic nevus': 'nevus',
            'dermatofibroma': 'df',
            'df': 'df',
            'seborrheic keratosis': 'seborrheic_keratosis',
            'queratose seborreica': 'seborrheic_keratosis',
            'queratose seborréica': 'seborrheic_keratosis',
            'sk': 'seborrheic_keratosis',
            'lentigo': 'lentigo',
            'lentigo solar': 'lentigo',
            'solar lentigo': 'lentigo',
            'vascular lesion': 'vasc',
            'lesão vascular': 'vasc',
            'lesao vascular': 'vasc',
            'vasc': 'vasc',
            'angioma': 'vasc',
            'hemangioma': 'vasc'
        }
        
        # Check API health
        try:
            self._check_api_health()
        except Exception as e:
            print(f"Error during API health check: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Check dataset existence
        try:
            self._check_dataset()
        except Exception as e:
            print(f"Error during dataset check: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def _check_api_health(self):
        print("Checking API health...")
        try:
            response = requests.get(f"{self.api_url}/ram", timeout=10)
            print(f"API response status: {response.status_code}")
            if response.status_code == 200:
                print("API is accessible")
            else:
                print(f"Warning: API responded with status {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: Cannot connect to API at {self.api_url}")
            print("Make sure the backend is running (docker compose up)!")
            raise e
        except requests.exceptions.Timeout as e:
            print(f"Timeout error: API did not respond within 10 seconds")
            raise e
        except Exception as e:
            print(f"Unexpected error connecting to API: {e}")
            raise e
    
    def _check_dataset(self):
        print("Checking dataset...")
        
        if not self.dataset_dir.exists():
            print(f"Error: Dataset directory not found: {self.dataset_dir}")
            print(f"Current working directory: {os.getcwd()}")
            print("Make sure ../test_dataset/ exists")
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        if not self.csv_path.exists():
            print(f"Error: CSV file not found: {self.csv_path}")
            print(f"Files in dataset directory: {list(self.dataset_dir.glob('*'))}")
            print("Make sure ../test_dataset/data.csv exists")
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        print(f"Dataset found at: {self.dataset_dir}")
        print(f"CSV file found at: {self.csv_path}")
    
    def normalize_clinical_impression(self, impression: str) -> str:
        if pd.isna(impression) or not isinstance(impression, str):
            return 'unknown'
        
        # Convert to lowercase and remove extra spaces
        impression_clean = impression.lower().strip()
        
        # Direct mapping check
        if impression_clean in self.clinical_to_dx_mapping:
            return self.clinical_to_dx_mapping[impression_clean]
        
        # Fuzzy matching for partial matches
        for clinical_term, dx_category in self.clinical_to_dx_mapping.items():
            if clinical_term in impression_clean or impression_clean in clinical_term:
                return dx_category
        
        # Pattern matching for common variations
        if any(term in impression_clean for term in ['melanoma', 'melanocítico maligno']):
            return 'melanoma'
        elif any(term in impression_clean for term in ['basal', 'bcc']):
            return 'bcc'
        elif any(term in impression_clean for term in ['escamosas', 'espinocelular', 'scc']):
            return 'scc'
        elif any(term in impression_clean for term in ['actínica', 'actinica', 'ak']):
            return 'ak'
        elif any(term in impression_clean for term in ['nevo', 'nevus', 'melanocítico benigno']):
            return 'nevus'
        elif any(term in impression_clean for term in ['dermatofibroma', 'df']):
            return 'df'
        elif any(term in impression_clean for term in ['seborreica', 'seborréica', 'sk']):
            return 'seborrheic_keratosis'
        elif any(term in impression_clean for term in ['lentigo', 'solar']):
            return 'lentigo'
        elif any(term in impression_clean for term in ['vascular', 'angioma', 'hemangioma']):
            return 'vasc'
        
        return 'unknown'
    
    def normalize_binary_classification(self, impression: str) -> str:
        dx_category = self.normalize_clinical_impression(impression)
        
        if dx_category == 'unknown':
            return 'unknown'
        
        # Malignant categories
        malignant_categories = ['melanoma', 'bcc', 'scc', 'ak']
        if dx_category in malignant_categories:
            return 'malignant'
        else:
            return 'benign'
    
    def load_test_data(self) -> pd.DataFrame:
        print("Loading test data...")
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} records from dataset")
            
            # Check required columns
            required_cols = ['image', 'binary', 'dx', 'clinical_impression_1', 'clinical_impression_2', 'clinical_impression_3']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            print(f"Available columns: {list(df.columns)}")
            
            if missing_cols:
                print(f"Error: Missing columns in CSV: {missing_cols}")
                raise ValueError(f"Missing columns in CSV: {missing_cols}")
            
            print("All required columns found")
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            traceback.print_exc()
            raise e
    
    def predict_image(self, image_name: str) -> Optional[Dict]:
        image_path = self.dataset_dir / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
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
                print(f"Warning: Prediction failed for {image_name}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            return None
    
    def create_analysis_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Creating analysis dataset...")
        analyses = []
        
        for _, row in df.iterrows():
            # Check each clinical impression column
            for col in ['clinical_impression_1', 'clinical_impression_2', 'clinical_impression_3']:
                clinical_impression = row[col]
                
                # Only create analysis if clinical impression exists and is not empty
                if pd.notna(clinical_impression) and str(clinical_impression).strip() != '':
                    analysis = {
                        'image': row['image'],
                        'ground_truth_binary': row['binary'],
                        'ground_truth_dx': row['dx'],
                        'clinical_impression': clinical_impression,
                        'ai_prediction': row.get('ai_prediction', None)
                    }
                    analyses.append(analysis)
        
        analysis_df = pd.DataFrame(analyses)
        print(f"Created {len(analysis_df)} individual analyses from {len(df)} images")
        return analysis_df
    
    def extract_isskin_predictions(self, analysis_df: pd.DataFrame) -> Tuple[List, List]:
        print("Extracting Isskin predictions...")
        isskin_binary_preds = []
        isskin_dx_preds = []
        
        for _, row in analysis_df.iterrows():
            if pd.isna(row['ai_prediction']) or row['ai_prediction'] is None:
                isskin_binary_preds.append('unknown')
                isskin_dx_preds.append('unknown')
            else:
                pred_data = row['ai_prediction']
                
                # Binary prediction - using same logic as standalone evaluator
                binary_pred = pred_data.get('binary_prediction', {})
                if 'malignant' in binary_pred and 'benign' in binary_pred:
                    # Convert percentages to probabilities (same as standalone evaluator)
                    malignant_prob = binary_pred['malignant'] / 100.0
                    benign_prob = binary_pred['benign'] / 100.0
                    
                    # Use same logic: 1 if malignant_prob > benign_prob else 0
                    pred_class = 1 if malignant_prob > benign_prob else 0
                    # Convert back to string for consistency with ground truth format
                    isskin_binary_preds.append('malignant' if pred_class == 1 else 'benign')
                else:
                    isskin_binary_preds.append('unknown')
                
                # DX prediction - using same logic as standalone evaluator
                dx_pred = pred_data.get('dx_prediction', {})
                if dx_pred:
                    predicted_class = max(dx_pred.keys(), key=dx_pred.get)
                    isskin_dx_preds.append(predicted_class)
                else:
                    isskin_dx_preds.append('unknown')
        
        print(f"Extracted {len(isskin_binary_preds)} binary predictions, {len(isskin_dx_preds)} dx predictions")
        return isskin_binary_preds, isskin_dx_preds
    
    def calculate_isskin_accuracy_binary(self, analysis_df: pd.DataFrame) -> float:
        # Convert ground truth to numeric values (same mapping as standalone evaluator)
        binary_mapping = {'malignant': 1, 'benign': 0}
        y_true = []
        y_pred = []
        
        for _, row in analysis_df.iterrows():
            ground_truth = row['ground_truth_binary']
            if ground_truth in binary_mapping:
                y_true.append(binary_mapping[ground_truth])
                
                # Extract prediction using same logic as standalone evaluator
                if pd.isna(row['ai_prediction']) or row['ai_prediction'] is None:
                    y_pred.append(0)  # assume benign (same as standalone evaluator)
                else:
                    pred_data = row['ai_prediction']
                    binary_pred = pred_data.get('binary_prediction', {})
                    
                    if 'malignant' in binary_pred and 'benign' in binary_pred:
                        malignant_prob = binary_pred['malignant'] / 100.0
                        benign_prob = binary_pred['benign'] / 100.0
                        pred_class = 1 if malignant_prob > benign_prob else 0
                        y_pred.append(pred_class)
                    else:
                        y_pred.append(0)  # assume benign
        
        if len(y_true) == 0:
            return 0.0
        
        return accuracy_score(y_true, y_pred)
    
    def calculate_isskin_accuracy_dx(self, analysis_df: pd.DataFrame) -> float:
        y_true = []
        y_pred = []
        
        for _, row in analysis_df.iterrows():
            ground_truth = row['ground_truth_dx']
            if pd.notna(ground_truth):
                # Extract prediction using same logic as standalone evaluator
                if pd.isna(row['ai_prediction']) or row['ai_prediction'] is None:
                    pred = 'unknown'
                else:
                    pred_data = row['ai_prediction']
                    dx_pred = pred_data.get('dx_prediction', {})
                    
                    if dx_pred:
                        pred = max(dx_pred.keys(), key=dx_pred.get)
                    else:
                        pred = 'unknown'
                
                # Only include cases where we made a valid prediction (same as standalone evaluator)
                if pred != 'unknown':
                    y_true.append(ground_truth)
                    y_pred.append(pred)
        
        if len(y_true) == 0:
            return 0.0
        
        return accuracy_score(y_true, y_pred)
    
    def calculate_agreement_and_accuracy(self, isskin_preds, doctor_preds, ground_truth, analysis_df, classification_type):
        # Calculate Isskin accuracy using the same logic as standalone evaluator
        if classification_type == 'binary':
            isskin_accuracy = self.calculate_isskin_accuracy_binary(analysis_df)
        else:  # dx
            isskin_accuracy = self.calculate_isskin_accuracy_dx(analysis_df)
        
        # Calculate Doctor accuracy (all cases where doctor made a prediction)
        doctor_valid_indices = []
        for i, (dp, gt) in enumerate(zip(doctor_preds, ground_truth)):
            if dp != 'unknown' and pd.notna(dp) and pd.notna(gt):
                doctor_valid_indices.append(i)
        
        if len(doctor_valid_indices) > 0:
            doctor_filtered = [doctor_preds[i] for i in doctor_valid_indices]
            gt_doctor_filtered = [ground_truth[i] for i in doctor_valid_indices]
            try:
                doctor_accuracy = accuracy_score(gt_doctor_filtered, doctor_filtered)
            except:
                doctor_accuracy = 0.0
        else:
            doctor_accuracy = 0.0
        
        # Calculate agreement (only cases where both made predictions)
        agreement_valid_indices = []
        for i, (ip, dp) in enumerate(zip(isskin_preds, doctor_preds)):
            if (ip != 'unknown' and dp != 'unknown' and 
                pd.notna(ip) and pd.notna(dp)):
                agreement_valid_indices.append(i)
        
        if len(agreement_valid_indices) > 0:
            isskin_agreement_filtered = [isskin_preds[i] for i in agreement_valid_indices]
            doctor_agreement_filtered = [doctor_preds[i] for i in agreement_valid_indices]
            
            # Agreement rate
            agreements = [ip == dp for ip, dp in zip(isskin_agreement_filtered, doctor_agreement_filtered)]
            agreement_rate = sum(agreements) / len(agreements)
            
            # Cohen's Kappa
            try:
                kappa = cohen_kappa_score(isskin_agreement_filtered, doctor_agreement_filtered)
            except:
                kappa = 0.0
        else:
            agreement_rate = 0.0
            kappa = 0.0
            agreements = []
        
        # Count total valid predictions for each method
        isskin_total_predictions = len([p for p in isskin_preds if p != 'unknown' and pd.notna(p)])
        doctor_total_predictions = len(doctor_valid_indices)
        
        return {
            'total_isskin_predictions': isskin_total_predictions,
            'total_doctor_predictions': doctor_total_predictions,
            'total_comparable_cases': len(agreement_valid_indices),
            'agreement_rate': agreement_rate,
            'disagreement_rate': 1 - agreement_rate,
            'cohen_kappa': kappa,
            'isskin_accuracy': isskin_accuracy,
            'doctor_accuracy': doctor_accuracy,
            'accuracy_difference': isskin_accuracy - doctor_accuracy,
            'agreements': len(agreements) - sum(agreements) if agreements else 0,
            'disagreements': sum(agreements) if agreements else 0
        }
    
    def compare_binary_classifications(self, analysis_df: pd.DataFrame) -> Dict:
        print("Comparing binary classifications...")
        
        ground_truth = analysis_df['ground_truth_binary'].tolist()
        isskin_preds, _ = self.extract_isskin_predictions(analysis_df)
        
        # Get doctor predictions
        doctor_preds = [
            self.normalize_binary_classification(impression) 
            for impression in analysis_df['clinical_impression'].tolist()
        ]
        
        # Calculate overall comparison
        comparison = self.calculate_agreement_and_accuracy(
            isskin_preds, 
            doctor_preds, 
            ground_truth,
            analysis_df,
            'binary'
        )
        
        return {
            'overall_comparison': comparison,
            'summary': {
                'total_analyses': len(analysis_df),
                'isskin_predictions': comparison.get('total_isskin_predictions', 0),
                'doctor_predictions': comparison.get('total_doctor_predictions', 0),
                'comparable_cases': comparison.get('total_comparable_cases', 0),
                'isskin_accuracy': comparison.get('isskin_accuracy', 0),
                'doctors_accuracy': comparison.get('doctor_accuracy', 0),
                'agreement_rate': comparison.get('agreement_rate', 0),
                'cohen_kappa': comparison.get('cohen_kappa', 0)
            }
        }
    
    def compare_dx_classifications(self, analysis_df: pd.DataFrame) -> Dict:
        print("Comparing DX classifications...")
        
        ground_truth = analysis_df['ground_truth_dx'].tolist()
        _, isskin_preds = self.extract_isskin_predictions(analysis_df)
        
        # Get doctor predictions
        doctor_preds = [
            self.normalize_clinical_impression(impression) 
            for impression in analysis_df['clinical_impression'].tolist()
        ]
        
        # Calculate overall comparison
        comparison = self.calculate_agreement_and_accuracy(
            isskin_preds,
            doctor_preds,
            ground_truth,
            analysis_df,
            'dx'
        )
        
        return {
            'overall_comparison': comparison,
            'summary': {
                'total_analyses': len(analysis_df),
                'isskin_predictions': comparison.get('total_isskin_predictions', 0),
                'doctor_predictions': comparison.get('total_doctor_predictions', 0),
                'comparable_cases': comparison.get('total_comparable_cases', 0),
                'isskin_accuracy': comparison.get('isskin_accuracy', 0),
                'doctors_accuracy': comparison.get('doctor_accuracy', 0),
                'agreement_rate': comparison.get('agreement_rate', 0),
                'cohen_kappa': comparison.get('cohen_kappa', 0)
            }
        }
    
    def run_comparison(self, output_file: Optional[str] = None) -> Dict:
        print("Starting Isskin vs Dermatologists comparison...")
        
        try:
            # Load data
            df = self.load_test_data()
            
            # Make Isskin predictions for all images
            print("Making Isskin predictions...")
            ai_predictions = []
            total_images = len(df)
            successful_predictions = 0
            
            for idx, row in df.iterrows():
                print(f"Processing image {idx + 1}/{total_images}: {row['image']}")
                prediction = self.predict_image(row['image'])
                ai_predictions.append(prediction)
                if prediction is not None:
                    successful_predictions += 1
            
            # Add AI predictions to DataFrame
            df['ai_prediction'] = ai_predictions
            
            print(f"Successfully predicted {successful_predictions}/{total_images} images")
            
            # Create analysis dataset (one row per analysis)
            analysis_df = self.create_analysis_dataset(df)
            
            # Compare binary classifications
            binary_comparison = self.compare_binary_classifications(analysis_df)
            
            # Compare dx classifications
            dx_comparison = self.compare_dx_classifications(analysis_df)
            
            # Compile results
            results = {
                'timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'total_images': len(df),
                    'total_analyses': len(analysis_df),
                    'successful_isskin_predictions': successful_predictions,
                    'failed_isskin_predictions': total_images - successful_predictions,
                    'isskin_success_rate': successful_predictions / total_images if total_images > 0 else 0
                },
                'binary_classification_comparison': binary_comparison,
                'dx_classification_comparison': dx_comparison
            }
            
            # Save results if requested
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"Results saved to: {output_path}")
            
            print("Comparison completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            traceback.print_exc()
            raise e


def main():
    print("Isskin vs Dermatologists Comparison Script")
    print("=" * 50)
    
    try:
        parser = argparse.ArgumentParser(description='Compare Isskin with Dermatologists')
        parser.add_argument('--api-url', default='http://localhost:8000', 
                           help='API URL (default: http://localhost:8000)')
        parser.add_argument('--output', help='File to save results (JSON)')
        
        args = parser.parse_args()
        
        print(f"Arguments: API URL = {args.api_url}, Output = {args.output}")
        
        # Create comparator
        comparator = IsskinvsDermatologistsComparison(api_url=args.api_url)
        
        # Execute comparison
        results = comparator.run_comparison(args.output)
        
        print("Comparison completed successfully!")
        if args.output:
            print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()