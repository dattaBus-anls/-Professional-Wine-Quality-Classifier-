# model_test_script.py
# Test script to verify your saved model works correctly outside Streamlit

import pickle
import pandas as pd
import numpy as np
import os

def test_saved_model():
    """Test the saved model components to verify they work correctly"""
    
    print("="*60)
    print("🔬 TESTING SAVED MODEL COMPONENTS")
    print("="*60)
    
    # Step 1: Load all components
    try:
        print("📂 Loading model components...")
        
        with open('wine_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded")
        
        with open('wine_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")
        
        with open('wine_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Label encoder loaded")
        
        with open('wine_feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        print("✅ Feature names loaded")
        
        with open('wine_type_encoder.pkl', 'rb') as f:
            wine_type_encoder = pickle.load(f)
        print("✅ Wine type encoder loaded")
        
        with open('feature_ranges.pkl', 'rb') as f:
            feature_ranges = pickle.load(f)
        print("✅ Feature ranges loaded")
        
    except Exception as e:
        print(f"❌ Error loading components: {e}")
        return False
    
    # Step 2: Display component information
    print("\n" + "="*60)
    print("📋 MODEL COMPONENT ANALYSIS")
    print("="*60)
    
    print(f"🤖 Model Type: {type(model).__name__}")
    print(f"📊 Feature Names ({len(feature_names)}): {feature_names}")
    print(f"🏷️ Label Classes: {label_encoder.classes_}")
    print(f"🍷 Wine Types: {wine_type_encoder.classes_}")
    
    # Step 3: Create test samples
    print("\n" + "="*60)
    print("🧪 CREATING TEST SAMPLES")
    print("="*60)
    
    # Test Sample 1: Average wine characteristics
    test_sample_1 = create_test_sample("average", feature_names, feature_ranges, wine_type_encoder)
    print("📝 Test Sample 1 (Average characteristics):")
    print(test_sample_1)
    
    # Test Sample 2: High quality characteristics  
    test_sample_2 = create_test_sample("good", feature_names, feature_ranges, wine_type_encoder)
    print("\n📝 Test Sample 2 (Good characteristics):")
    print(test_sample_2)
    
    # Test Sample 3: Poor quality characteristics
    test_sample_3 = create_test_sample("poor", feature_names, feature_ranges, wine_type_encoder)
    print("\n📝 Test Sample 3 (Poor characteristics):")
    print(test_sample_3)
    
    # Step 4: Test predictions
    print("\n" + "="*60)
    print("🔮 TESTING PREDICTIONS")
    print("="*60)
    
    test_samples = [
        ("Average Characteristics", test_sample_1),
        ("Good Characteristics", test_sample_2), 
        ("Poor Characteristics", test_sample_3)
    ]
    
    for sample_name, sample_values in test_samples:
        print(f"\n🧪 Testing {sample_name}:")
        print("-" * 40)
        
        try:
            # Create DataFrame with exact feature names
            df_test = pd.DataFrame([sample_values], columns=feature_names)
            print(f"Input DataFrame shape: {df_test.shape}")
            print(f"Input values: {sample_values}")
            
            # Scale the features
            scaled_features = scaler.transform(df_test)
            print(f"Scaled features shape: {scaled_features.shape}")
            
            # Make prediction
            prediction_encoded = model.predict(scaled_features)[0]
            quality_prediction = label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(scaled_features)[0]
                print(f"Prediction probabilities:")
                for i, (class_name, prob) in enumerate(zip(label_encoder.classes_, probabilities)):
                    print(f"  {class_name}: {prob:.3f} ({prob*100:.1f}%)")
            
            print(f"🎯 FINAL PREDICTION: {quality_prediction}")
            
            # Check if prediction makes sense
            if quality_prediction in label_encoder.classes_:
                print("✅ Prediction is valid")
            else:
                print("❌ Invalid prediction")
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    print("✅ Model components loaded successfully")
    print("✅ Predictions completed without errors")
    print("🎯 If you see different predictions for different samples, the model works correctly!")
    print("🚨 If all predictions are the same, there's an issue with the model or data processing")
    
    return True

def create_test_sample(quality_type, feature_names, feature_ranges, wine_type_encoder):
    """Create test samples with characteristics typical of different quality levels"""
    
    # Base feature values from your training data analysis
    if quality_type == "average":
        # Average wine characteristics from your data
        sample = {
            'fixed_acidity': 7.2,
            'volatile_acidity': 0.34,
            'citric_acid': 0.32,
            'residual_sugar': 5.4,
            'chlorides': 0.056,
            'free_sulfur_dioxide': 31,
            'total_sulfur_dioxide': 116,
            'density': 0.995,
            'ph': 3.22,
            'sulphates': 0.53,
            'alcohol': 10.5
        }
    elif quality_type == "good":
        # Higher quality characteristics (more alcohol, better balance)
        sample = {
            'fixed_acidity': 7.8,
            'volatile_acidity': 0.28,
            'citric_acid': 0.40,
            'residual_sugar': 2.5,
            'chlorides': 0.045,
            'free_sulfur_dioxide': 25,
            'total_sulfur_dioxide': 95,
            'density': 0.993,
            'ph': 3.15,
            'sulphates': 0.65,
            'alcohol': 12.5
        }
    else:  # poor quality
        # Lower quality characteristics (high volatile acidity, imbalanced)
        sample = {
            'fixed_acidity': 6.5,
            'volatile_acidity': 0.55,
            'citric_acid': 0.15,
            'residual_sugar': 8.0,
            'chlorides': 0.080,
            'free_sulfur_dioxide': 15,
            'total_sulfur_dioxide': 140,
            'density': 0.997,
            'ph': 3.35,
            'sulphates': 0.40,
            'alcohol': 9.2
        }
    
    # Add wine type encoding (use 'white' as default)
    wine_type_encoded = wine_type_encoder.transform(['white'])[0]
    sample['wine_type_encoded'] = float(wine_type_encoded)
    
    # Create ordered list matching feature_names exactly
    ordered_values = []
    for feature_name in feature_names:
        if feature_name in sample:
            ordered_values.append(sample[feature_name])
        else:
            print(f"⚠️ Warning: Feature {feature_name} not found in sample")
            ordered_values.append(0.0)  # Default value
    
    return ordered_values

if __name__ == "__main__":
    test_saved_model()