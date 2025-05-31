# train_model_and_save.py
import pandas as pd
import re
import nltk # Import nltk
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer # Uncomment if you used lemmatization

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuration ---
# Define these carefully to match your successful notebook experiment
# TF-IDF Parameters (EXAMPLE - use your best ones)
TFIDF_MAX_DF = 0.75
TFIDF_MIN_DF = 5
TFIDF_NGRAM_RANGE = (1, 2) # Use (1,1) if you only used unigrams

# Logistic Regression Parameters (EXAMPLE - use your best ones)
LOGREG_C = 1.0
LOGREG_SOLVER = 'liblinear'
LOGREG_PENALTY = 'l2'
LOGREG_RANDOM_STATE = 42
LOGREG_MAX_ITER = 1000 # Increase if solver has convergence issues

# Label Mapping (CRITICAL - ensure this matches how you will interpret in Django)
# FAKE=0, REAL=1 (This is a common convention)
LABEL_MAPPING = {'FAKE': 0, 'REAL': 1}
# POSITIVE_CLASS_LABEL_NUMERIC = 1 # The numeric label representing "REAL"

# Data file paths (update if your paths are different)
FAKE_NEWS_PATH = '/Users/harshit/Downloads/archive (1)/Fake.csv'
TRUE_NEWS_PATH = '/Users/harshit/Downloads/archive (1)/True.csv'
COMBINED_NEWS_CSV = 'news.csv' # Temporary combined file

TEXT_COLUMN = 'text'        # Column containing the news article body after combining
LABEL_COLUMN = 'label'      # Column containing 'FAKE' or 'REAL' labels

OUTPUT_MODEL_DIR = os.path.join('detector_app', 'ml_model') # Django app's model folder
MODEL_FILENAME = 'model.joblib'
VECTORIZER_FILENAME = 'vectorizer.joblib'
PREPROCESSING_UTILS_FILENAME = 'preprocessing_utils.py'


# --- Download NLTK Resources ---
def download_nltk_resources():
    try:
        stopwords.words('english')
        print("NLTK stopwords already available.")
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords')
        print("NLTK stopwords downloaded.")

    # Uncomment if you use WordNetLemmatizer
    # try:
    #     # A simple test to see if wordnet is available
    #     WordNetLemmatizer().lemmatize("test")
    #     print("NLTK WordNet already available.")
    # except LookupError:
    #     print("NLTK WordNet not found. Downloading...")
    #     nltk.download('wordnet')
    #     print("NLTK WordNet downloaded.")

download_nltk_resources() # Call this early

# --- 1. Preprocessing Function ---
# This function will be saved for the Django app.
# Ensure any resources it uses (like stop_words) are defined globally or passed if needed.
stop_words_set = set(stopwords.words('english')) # Define it once globally for this script
# lemmatizer = WordNetLemmatizer() # Uncomment if used

def preprocess_text_for_model(text_input): # Renamed for clarity within this script
    if not isinstance(text_input, str):
        return ""
    text = text_input.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    words = text.split()
    words = [word for word in words if word not in stop_words_set]
    # words = [lemmatizer.lemmatize(word) for word in words] # Uncomment if used
    return " ".join(words)

# --- Main Training Script ---
def create_combined_dataset_and_train():
    print("--- Starting Dataset Creation and Model Training ---")

    # --- A. Create Combined Dataset (from your original script) ---
    print(f"Loading FAKE news from: {FAKE_NEWS_PATH}")
    try:
        fake_df = pd.read_csv(FAKE_NEWS_PATH)
        print(f"Loading TRUE news from: {TRUE_NEWS_PATH}")
        true_df = pd.read_csv(TRUE_NEWS_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not load Fake.csv or True.csv. Please check paths. {e}")
        return

    fake_df['label'] = 'FAKE' # Assign label before selecting columns
    true_df['label'] = 'REAL' # Assign label before selecting columns

    # Combine them
    df_combined = pd.concat([fake_df, true_df], ignore_index=True)

    # Shuffle the dataset
    df_combined = df_combined.sample(frac=1, random_state=LOGREG_RANDOM_STATE).reset_index(drop=True)

    # Select relevant columns (text and label)
    # Check if 'text' column exists, if not, maybe 'title' or other columns exist.
    # For this dataset structure, 'text' is usually present.
    if TEXT_COLUMN not in df_combined.columns:
        print(f"Error: Expected text column '{TEXT_COLUMN}' not found in the combined dataframe.")
        print(f"Available columns: {df_combined.columns.tolist()}")
        # You might want to inspect df_combined.head() here if this error occurs
        return
    
    df = df_combined[[TEXT_COLUMN, LABEL_COLUMN]].copy() # Use .copy() to avoid SettingWithCopyWarning
    df.dropna(subset=[TEXT_COLUMN], inplace=True) # Remove rows where text is missing
    print(f"Combined dataset shape: {df.shape}")
    df.to_csv(COMBINED_NEWS_CSV, index=False)
    print(f"Combined '{COMBINED_NEWS_CSV}' created successfully.")

    # --- B. Now proceed with training using the combined 'news.csv' ---

    # --- 2. Load Data (the combined one we just created) ---
    print(f"Loading combined data from '{COMBINED_NEWS_CSV}' for training...")
    try:
        df = pd.read_csv(COMBINED_NEWS_CSV)
    except FileNotFoundError:
        print(f"Error: {COMBINED_NEWS_CSV} not found. Should have been created above.")
        return

    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    df = df[df[TEXT_COLUMN].astype(str).str.strip().astype(bool)] # Ensure text is not just spaces
    print(f"Shape after loading and initial NA drop: {df.shape}")


    # --- 3. Apply Preprocessing ---
    print("Applying text preprocessing...")
    df['processed_text'] = df[TEXT_COLUMN].apply(preprocess_text_for_model)
    df['label_encoded'] = df[LABEL_COLUMN].map(LABEL_MAPPING)

    # Filter out rows that became empty after processing or had unmapped labels
    df.dropna(subset=['processed_text', 'label_encoded'], inplace=True)
    df = df[df['processed_text'].str.strip().astype(bool)]
    df['label_encoded'] = df['label_encoded'].astype(int) # Ensure integer type
    print(f"Shape after preprocessing and label encoding: {df.shape}")

    if df.empty or len(df['label_encoded'].unique()) < 2:
        print("Error: Not enough data or classes after preprocessing. Exiting.")
        return

    X = df['processed_text']
    y = df['label_encoded']
    print(f"Class distribution for training:\n{y.value_counts(normalize=True)}")

    # --- 4. Train-Test Split ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=LOGREG_RANDOM_STATE, stratify=y
    )
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Not enough data for train/test split. Ensure dataset is large enough.")
        return
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- 5. TF-IDF Vectorization ---
    print("Initializing and fitting TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words=None # Stop words are handled in preprocess_text_for_model
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF Matrix shape (Train): {X_train_tfidf.shape}")
    print(f"Number of features (vocabulary size): {len(vectorizer.get_feature_names_out())}")

    # --- 6. Model Training (Logistic Regression) ---
    print("Initializing and training Logistic Regression model...")
    model = LogisticRegression(
        C=LOGREG_C,
        solver=LOGREG_SOLVER,
        penalty=LOGREG_PENALTY,
        random_state=LOGREG_RANDOM_STATE,
        max_iter=LOGREG_MAX_ITER
    )
    model.fit(X_train_tfidf, y_train)

    # --- 7. Evaluate ---
    print("Evaluating model on the test set...")
    y_pred_test = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Set Accuracy: {accuracy*100:.2f}%")
    # Generate target names from the LABEL_MAPPING for the report
    target_names_for_report = [name for name, num in sorted(LABEL_MAPPING.items(), key=lambda item: item[1])]
    print("\nTest Set Classification Report:\n", classification_report(y_test, y_pred_test, target_names=target_names_for_report))


    # --- 8. Save the Model and Vectorizer ---
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_MODEL_DIR, MODEL_FILENAME)
    vectorizer_path = os.path.join(OUTPUT_MODEL_DIR, VECTORIZER_FILENAME)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

    # --- 9. Save the preprocessing function for Django app ---
    # The function saved here MUST be identical to preprocess_text_for_model
    # and use the same global `stop_words_set` or have it defined within.
    preprocessing_utils_path = os.path.join(OUTPUT_MODEL_DIR, PREPROCESSING_UTILS_FILENAME)
    with open(preprocessing_utils_path, 'w') as f:
        f.write("import re\n")
        f.write("from nltk.corpus import stopwords\n")
        f.write("# from nltk.stem import WordNetLemmatizer # Uncomment if you used it\n")
        f.write("\n")
        f.write("# This 'stop_words_set' name must match what's used in the function below\n")
        f.write("stop_words_set = set(stopwords.words('english'))\n")
        f.write("# lemmatizer = WordNetLemmatizer() # Uncomment if you used it\n")
        f.write("\n")
        f.write("# Name this function consistently for the Django app to import\n")
        f.write("def preprocess_text_for_prediction(text_input):\n")
        f.write("    if not isinstance(text_input, str):\n")
        f.write("        return \"\"\n")
        f.write("    text = text_input.lower()\n")
        f.write("    text = re.sub(r'\\[.*?\\]', '', text) # Note: Escaped backslash for regex string literal\n")
        f.write("    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)\n")
        f.write("    text = re.sub(r'<.*?>+', '', text)\n")
        f.write("    text = re.sub(r'[^a-z\\s]', '', text)\n")
        f.write("    text = re.sub(r'\\n', ' ', text)\n")
        f.write("    text = re.sub(r'\\w*\\d\\w*', '', text)\n")
        f.write("    words = text.split()\n")
        f.write("    words = [word for word in words if word not in stop_words_set]\n")
        f.write("    # words = [lemmatizer.lemmatize(word) for word in words] # Uncomment if used\n")
        f.write("    return \" \".join(words)\n")
    print(f"Preprocessing utility saved to: {preprocessing_utils_path}")
    print("--- Dataset Creation and Training Process Completed ---")

if __name__ == "__main__":
    create_combined_dataset_and_train()