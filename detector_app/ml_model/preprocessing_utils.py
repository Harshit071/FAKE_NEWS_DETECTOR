import re
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer # Uncomment if you used it

# This 'stop_words_set' name must match what's used in the function below
stop_words_set = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer() # Uncomment if you used it

# Name this function consistently for the Django app to import
def preprocess_text_for_prediction(text_input):
    if not isinstance(text_input, str):
        return ""
    text = text_input.lower()
    text = re.sub(r'\[.*?\]', '', text) # Note: Escaped backslash for regex string literal
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words_set]
    # words = [lemmatizer.lemmatize(word) for word in words] # Uncomment if used
    return " ".join(words)
