import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import InconsistentVersionWarning

def run_demo():
    print("🚀 ML PROJECT - MODEL INFERENCE DEMO")
    print("=" * 50)
    
    try:
        # Load the FINAL processed test data
        print("📂 Loading FINAL processed test data...")
        test_data = pd.read_csv('data/test_data_final.csv')
        X_test = test_data.drop(columns=['diabetic'])
        y_test = test_data['diabetic']
        
        print(f"   - Test samples: {X_test.shape[0]}")
        print(f"   - Features: {X_test.shape[1]}")
        print(f"   - Feature names: {list(X_test.columns)}")
        
        # Load models
        print("📂 Loading models...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="pickle")
            
            scaler = joblib.load('models/scaler.pkl')
            models = {
                'Logistic Regression': joblib.load('models/Logistic_Regression_best.pkl'),
                'Random Forest': joblib.load('models/Random_Forest_best.pkl'),
                'XGBoost': joblib.load('models/XGBoost_best.pkl')
            }
        
        print("✅ All models loaded successfully!")
        
        # Scale the test data (using the same scaler from training)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n🎯 MODEL PERFORMANCE ON TEST SET:")
        print("=" * 50)
        
        results = []
        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = (y_pred == y_test).mean()
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)  
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            results.append({
                'Model': name, 
                'AUC-ROC': auc, 
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            print(f"\n📊 {name}:")
            print(f"   - AUC-ROC: {auc:.4f}")
            print(f"   - Accuracy: {accuracy:.4f}")
            print(f"   - Precision: {precision:.4f}")
            print(f"   - Recall: {recall:.4f}")
            print(f"   - F1-Score: {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"   - Confusion Matrix:")
            print(f"        TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
            print(f"        FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")
            
            # Medical impact analysis
            fn_rate = cm[1,0] / (cm[1,0] + cm[1,1])  # False Negative Rate
            print(f"   - Medical Impact:")
            print(f"        False Negative Rate: {fn_rate:.4f} ({cm[1,0]} missed diabetes cases)")
        
        # Show comparison table
        print("\n" + "=" * 60)
        print("📈 MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Best model by AUC-ROC
        best_auc_model = results_df.iloc[0]
        best_auc_name = best_auc_model['Model']
        
        # Best model by Recall (most important for medical)
        best_recall_model = results_df.loc[results_df['Recall'].idxmax()]
        best_recall_name = best_recall_model['Model']
        best_recall_value = best_recall_model['Recall']
        
        print(f"\n🏆 BEST MODEL BY AUC-ROC: {best_auc_name}")
        print(f"   - Test AUC-ROC: {best_auc_model['AUC-ROC']:.4f}")
        
        print(f"\n⭐ BEST MODEL BY RECALL (Medical Priority): {best_recall_name}")
        print(f"   - Recall: {best_recall_value:.4f}")
        print(f"   - Detects {best_recall_value*100:.1f}% of actual diabetes cases")
        
        # Quick demo with sample data
        print("\n" + "=" * 50)
        print("🔍 SINGLE SAMPLE PREDICTION DEMO")
        print("=" * 50)
        
        sample_data = pd.read_csv('data/test_sample_final.csv')
        X_sample = sample_data.drop(columns=['diabetic'])
        y_sample = sample_data['diabetic']
        X_sample_scaled = scaler.transform(X_sample)
        
        # Use the best model by recall for medical safety
        best_model = models[best_recall_name]
        
        print(f"Using {best_recall_name} (best recall model) for predictions:\n")
        
        for i in range(len(X_sample)):
            # Get prediction
            prediction = best_model.predict(X_sample_scaled[i:i+1])[0]
            probability = best_model.predict_proba(X_sample_scaled[i:i+1])[0]
            
            # Convert to readable labels
            actual_label = "Diabetes" if y_sample.iloc[i] == 1 else "No Diabetes"
            predicted_label = "Diabetes" if prediction == 1 else "No Diabetes"
            prob_diabetes = probability[1]
            
            # Check if correct
            is_correct = "✅ CORRECT" if prediction == y_sample.iloc[i] else "❌ WRONG"
            
            print(f"Sample {i+1}:")
            print(f"   Actual: {actual_label}")
            print(f"   Predicted: {predicted_label} (probability: {prob_diabetes:.3f})")
            print(f"   Result: {is_correct}")
            
            # Medical risk assessment
            if actual_label == "Diabetes" and predicted_label == "No Diabetes":
                print(f"   ⚠️  HIGH RISK: Missed diabetes case!")
            elif actual_label == "No Diabetes" and predicted_label == "Diabetes":
                print(f"   ℹ️  False alarm - requires follow-up test")
            print()
        
        # Final medical recommendation
        print("=" * 50)
        print("🏥 MEDICAL RECOMMENDATION")
        print("=" * 50)
        print(f"Recommended model: {best_recall_name}")
        print(f"Reason: Highest recall ({best_recall_value:.4f}) - minimizes missed diabetes cases")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()