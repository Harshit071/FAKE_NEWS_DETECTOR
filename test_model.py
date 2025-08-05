import joblib
import pandas as pd
from detector_app.ml_model.preprocessing_utils import preprocess_text_for_prediction

# Load model and vectorizer
model = joblib.load('detector_app/ml_model/model.joblib')
vectorizer = joblib.load('detector_app/ml_model/vectorizer.joblib')

print("Model classes:", model.classes_)

# Test examples - more diverse
test_cases = [
    ("BREAKING: Scientists discover that 5G networks are actually mind control devices installed by the government to control population behavior.", "FAKE"),
    ("WASHINGTON (Reuters) - The Federal Reserve raised interest rates by 25 basis points on Wednesday, citing continued economic growth and inflation concerns.", "REAL"),
    ("NASA's Perseverance rover successfully landed on Mars on February 18, 2021, at 3:55 p.m. EST.", "REAL"),
    ("SHOCKING: Tom Hanks arrested for human trafficking in Hollywood! Multiple sources confirm that the beloved actor was taken into custody.", "FAKE"),
    ("LONDON (Reuters) - The Bank of England kept interest rates unchanged at 0.1% on Thursday, as expected, but signaled it was ready to act if the economic recovery falters.", "REAL"),
    ("URGENT: New study proves that drinking bleach cures all diseases including cancer! Doctors are hiding this miracle cure from the public.", "FAKE"),
    ("PARIS (Reuters) - French President Emmanuel Macron announced new climate initiatives on Tuesday, pledging to reduce carbon emissions by 40% by 2030.", "REAL"),
    ("CONSPIRACY ALERT: The moon landing was completely faked! New evidence shows NASA never went to space.", "FAKE")
]

correct = 0
total = len(test_cases)

for text, expected_label in test_cases:
    print(f"\n--- Testing: {expected_label} ---")
    print(f"Text: {text[:50]}...")
    
    # Preprocess
    processed = preprocess_text_for_prediction(text)
    print(f"Processed: {processed[:50]}...")
    
    # Vectorize
    vector = vectorizer.transform([processed])
    
    # Predict
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    
    predicted_label = 'FAKE' if pred == 0 else 'REAL'
    is_correct = predicted_label == expected_label
    if is_correct:
        correct += 1
    
    print(f"Prediction: {pred} (0=FAKE, 1=REAL)")
    print(f"Probabilities: FAKE={proba[0]:.3f}, REAL={proba[1]:.3f}")
    print(f"Expected: {expected_label}, Got: {predicted_label}")
    print(f"Correct: {is_correct}")

print(f"\n=== SUMMARY ===")
print(f"Total tests: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {correct/total*100:.1f}%") 