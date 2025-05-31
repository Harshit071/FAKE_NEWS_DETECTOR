# detector_app/views.py
from django.shortcuts import render
from django.conf import settings # To build file paths using BASE_DIR
import joblib
import os
import numpy as np # For more robust probability handling

# Import your form
from .forms import NewsTextForm

# Import your specific preprocessing function from the ml_model package
# This function name should match what is saved by train_model_and_save.py
try:
    from .ml_model.preprocessing_utils import preprocess_text_for_prediction
    PREPROCESSING_FUNCTION_LOADED = True
except ImportError:
    PREPROCESSING_FUNCTION_LOADED = False
    print("🚨 CRITICAL ERROR: Could not import 'preprocess_text_for_prediction' from .ml_model.preprocessing_utils")
    print("Ensure 'detector_app/ml_model/preprocessing_utils.py' exists and is correct.")


# --- Configuration for Prediction Interpretation ---
# This MUST align with LABEL_MAPPING in your train_model_and_save.py script
# Example: If train_script had LABEL_MAPPING = {'FAKE': 0, 'REAL': 1}
LABEL_NUMERIC_TO_STRING = {
    0: "FAKE",  # Assuming 0 was for FAKE news
    1: "REAL"   # Assuming 1 was for REAL news
}
# POSITIVE_CLASS_NUMERIC_LABEL = 1 # The numeric label that corresponds to "REAL" or your positive class

# --- Load Model and Vectorizer ONCE when Django starts ---
MODEL_FILENAME = 'model.joblib' # Ensure this matches the filename used in train_model_and_save.py
VECTORIZER_FILENAME = 'vectorizer.joblib' # Ensure this matches

MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector_app', 'ml_model', MODEL_FILENAME)
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'detector_app', 'ml_model', VECTORIZER_FILENAME)

model = None
vectorizer = None
critical_load_error_message = None # For errors preventing prediction

try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Attempting to load vectorizer from: {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and Vectorizer loaded successfully!")

    # Optional: Print some info about the loaded components
    if hasattr(vectorizer, 'get_feature_names_out'):
        print(f"   Vectorizer vocabulary size: {len(vectorizer.get_feature_names_out())}")
    elif hasattr(vectorizer, 'vocabulary_'): # Older scikit-learn
        print(f"   Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
    if hasattr(model, 'classes_'):
        print(f"   Model classes: {model.classes_}")

except FileNotFoundError as fnf_error:
    critical_load_error_message = (
        f"🚨 CRITICAL ERROR: Model or Vectorizer file not found. {fnf_error}. "
        f"Please ensure '{MODEL_FILENAME}' and '{VECTORIZER_FILENAME}' are in "
        f"'detector_app/ml_model/' and the training script (train_model_and_save.py) has been run successfully."
    )
    print(critical_load_error_message)
except Exception as e:
    critical_load_error_message = f"🚨 CRITICAL ERROR: An unexpected error occurred loading the model/vectorizer: {e}"
    print(critical_load_error_message)

if not PREPROCESSING_FUNCTION_LOADED and not critical_load_error_message:
    critical_load_error_message = "🚨 CRITICAL ERROR: Preprocessing function could not be loaded."


def predict_news_view(request): # Renamed view function for clarity
    form = NewsTextForm()
    prediction_display_text = None # For the final message like "The news is FAKE (Conf: 80%)"
    prediction_class_name_for_css = None # To store 'REAL' or 'FAKE' for CSS styling
    submitted_text_for_template = None # To show the user what they submitted
    # debug_info = {} # Optional: If you want to pass detailed debug info to the template

    # Handle critical loading errors that prevent any prediction
    if critical_load_error_message:
        return render(request, 'detector_app/index.html', {
            'form': form,
            'error_message': critical_load_error_message
        })
    
    # Fallback if model/vectorizer objects are None (should be caught by critical_load_error_message)
    if not model or not vectorizer or not PREPROCESSING_FUNCTION_LOADED:
        return render(request, 'detector_app/index.html', {
            'form': form,
            'error_message': "Machine learning components are not fully available. Please contact admin."
        })

    if request.method == 'POST':
        form = NewsTextForm(request.POST)
        if form.is_valid():
            news_text_input = form.cleaned_data['news_article']
            submitted_text_for_template = news_text_input

            print(f"\n--- New Prediction Request ---")
            print(f"DEBUG: Original Text from UI (first 200 chars): '{news_text_input[:200]}...'")
            # debug_info['original_text'] = news_text_input

            # 1. Preprocess the input text using the imported function
            processed_text = preprocess_text_for_prediction(news_text_input)
            print(f"DEBUG: Text after preprocessing (first 200 chars): '{processed_text[:200]}...'")
            # debug_info['processed_text'] = processed_text

            if not processed_text.strip(): # If text is empty after preprocessing
                 prediction_display_text = "Input text is empty or contains only common/special characters after processing. Cannot predict."
                 prediction_class_name_for_css = "INVALID" # A CSS class for invalid input
            else:
                try:
                    # 2. Vectorize the preprocessed text
                    # The vectorizer expects a list of documents, so pass [processed_text]
                    text_vector = vectorizer.transform([processed_text])
                    print(f"DEBUG: Shape of text_vector: {text_vector.shape}")
                    # debug_info['vector_shape'] = str(text_vector.shape)
                    
                    if text_vector.nnz == 0 : # Number of non-zero elements in the sparse matrix
                        print("⚠️ WARNING: Text vector is all zeros. Input may not contain any known vocabulary words.")
                        # debug_info['vector_status'] = "All zeros (likely only Out-Of-Vocabulary words)"


                    # 3. Make prediction
                    # model.predict() returns an array, e.g. [0] or [1]
                    prediction_numeric_array = model.predict(text_vector)
                    prediction_numeric = prediction_numeric_array[0] # Get the single prediction value
                    print(f"DEBUG: Numeric prediction from model: {prediction_numeric} (Type: {type(prediction_numeric)})")
                    # debug_info['numeric_prediction'] = prediction_numeric

                    # Get probabilities: model.predict_proba() returns array like [[prob_class_0, prob_class_1]]
                    prediction_probabilities_array = model.predict_proba(text_vector)
                    prediction_probabilities = prediction_probabilities_array[0] # Probabilities for the first (and only) sample
                    print(f"DEBUG: Prediction probabilities (for classes {model.classes_ if hasattr(model, 'classes_') else 'N/A'}): {prediction_probabilities}")
                    # debug_info['probabilities'] = str(prediction_probabilities)

                    # 4. Interpret prediction
                    predicted_label_numeric_int = int(prediction_numeric) # Ensure it's an int for dict lookup
                    
                    # Get the string name of the predicted class (e.g., "FAKE" or "REAL")
                    prediction_class_name_for_css = LABEL_NUMERIC_TO_STRING.get(predicted_label_numeric_int, "UNKNOWN")
                    
                    # Determine confidence for the *predicted* class
                    confidence_value = 0.0
                    if hasattr(model, 'classes_'):
                        try:
                            # Find the index of the predicted class in model.classes_
                            # model.classes_ might be [0 1] or [1 0] or other orders if not binary
                            class_index_in_model = np.where(model.classes_ == predicted_label_numeric_int)[0][0]
                            confidence_value = prediction_probabilities[class_index_in_model] * 100
                        except (IndexError, ValueError) as e:
                            print(f"⚠️ WARNING: Could not accurately determine confidence due to class index issue with model.classes_ ({model.classes_}): {e}")
                            # Fallback: If classes_ is simply [0, 1] and matches our numeric label
                            if predicted_label_numeric_int < len(prediction_probabilities):
                                confidence_value = prediction_probabilities[predicted_label_numeric_int] * 100
                            else:
                                confidence_value = 50.0 # Unable to determine, assign neutral
                    else:
                        # Fallback if model.classes_ is not available (less common for sklearn classifiers)
                         if predicted_label_numeric_int < len(prediction_probabilities):
                                confidence_value = prediction_probabilities[predicted_label_numeric_int] * 100
                         else:
                            confidence_value = 50.0

                    print(f"DEBUG: Predicted Class Name: {prediction_class_name_for_css}, Confidence: {confidence_value:.2f}%")
                    prediction_display_text = f"The news is predicted to be: {prediction_class_name_for_css} (Confidence: {confidence_value:.2f}%)"

                except Exception as e:
                    print(f"🚨 ERROR during prediction pipeline: {e}")
                    # Log the full traceback for detailed debugging if needed:
                    # import traceback
                    # print(traceback.format_exc())
                    prediction_display_text = f"An error occurred while trying to predict: {e}"
                    prediction_class_name_for_css = "ERROR" # A CSS class for error display
                    # debug_info['prediction_error'] = str(e)

    # Prepare context for the template
    context = {
        'form': form,
        'prediction_text': prediction_display_text,
        'prediction_label': prediction_class_name_for_css, # Used for CSS class in template
        'submitted_text': submitted_text_for_template,
        # 'debug_info': debug_info # Uncomment if you want to display debug_info in the template
    }
    return render(request, 'detector_app/index.html', context)