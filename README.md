# Fake News Detector

This project is a Fake News Detection system built using Natural Language Processing (NLP) techniques and Machine Learning. It features a web interface built with Django that allows users to input news text and receive a prediction of whether the news is REAL or FAKE.

The core ML pipeline uses a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text into numerical features and a Logistic Regression classifier. The model was trained on a combination of 'Fake.csv' and 'True.csv' (datasets often found on platforms like Kaggle) and achieved approximately 94% accuracy on the test set after extensive preprocessing, EDA, and hyperparameter tuning.

**Note:** This repository uses Git Large File Storage (LFS) to manage large data files like `news.csv`.

## Features

*   Classifies news articles as REAL or FAKE.
*   Web interface for easy user interaction.
*   Backend powered by Django.
*   Machine Learning model using Scikit-learn (TF-IDF + Logistic Regression).
*   Provides a confidence score for the prediction.

## Technologies Used

*   **Backend:** Python, Django
*   **Machine Learning:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`)
*   **NLP Preprocessing:** NLTK (stopwords), Regex
*   **Data Handling:** Pandas
*   **Large File Storage:** Git LFS
*   **Development Environment:** Jupyter Notebook (for initial model development), VS Code (or your editor)
*   **Version Control:** Git, GitHub

## Project Structure

fakenews_project/
├── manage.py
├── fakenews_project/ # Django project settings
├── detector_app/ # Django app for the detector
│ ├── migrations/
│ ├── static/
│ ├── templates/
│ ├── ml_model/ # Trained model, vectorizer, preprocessing_utils.py
│ │ ├── model.joblib
│ │ ├── vectorizer.joblib
│ │ └── preprocessing_utils.py
│ ├── forms.py
│ ├── views.py
│ ├── urls.py
│ └── ... (other app files)
├── train_model_and_save.py # Script to preprocess data, train, and save the ML model
├── news.csv # Combined dataset (tracked with Git LFS)
├── .gitattributes # Configures Git LFS tracking
├── README.md
└── requirements.txt # Python package dependencies


## Setup and Installation

1.  **Prerequisites:**
    *   Ensure Git is installed.
    *   **Install Git LFS:** Follow the instructions at [https://git-lfs.github.com/](https://git-lfs.github.com/). After installation, run `git lfs install` once to initialize Git LFS for your user.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/Harshit071/FAKE_NEWS_DETECTOR.git
    cd FAKE_NEWS_DETECTOR
    ```
    *(Git LFS should automatically download the large `news.csv` file during the clone if LFS is installed correctly).*

3.  **Create and activate a virtual environment:**
    *   Using Conda:
        ```bash
        conda create --name fakenews_env python=3.9  # Or your preferred version
        conda activate fakenews_env
        ```
    *   Using venv:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate  # On macOS/Linux
        # .venv\Scripts\activate    # On Windows
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK resources (if not handled by `train_model_and_save.py`'s initial run):**
    Open a Python interpreter within your active environment:
    ```python
    import nltk
    nltk.download('stopwords')
    # nltk.download('wordnet') # If lemmatization was used
    exit()
    ```
    (The provided `train_model_and_save.py` script attempts to download stopwords if missing).

6.  **Dataset `news.csv`:**
    *   This file is tracked using Git LFS and should have been downloaded when you cloned the repository (if Git LFS was installed).
    *   Alternatively, if you have the original `Fake.csv` and `True.csv` files, you can place them according to the paths in `train_model_and_save.py` and the script will regenerate `news.csv`.

7.  **Train the model and generate artifacts (Optional if `news.csv`, `model.joblib`, `vectorizer.joblib` are already up-to-date in the repo):**
    If you need to retrain or if model artifacts are not included/up-to-date:
    (Ensure paths in `train_model_and_save.py` are correct for your dataset location if regenerating `news.csv`)
    ```bash
    python train_model_and_save.py
    ```
    This will create/update `model.joblib`, `vectorizer.joblib`, and `preprocessing_utils.py` in `detector_app/ml_model/`.

8.  **Run Django migrations:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

9.  **Start the Django development server:**
    ```bash
    python manage.py runserver
    ```
    The application will be available at `http://127.0.0.1:8080/`.

## Usage

1.  Navigate to `http://127.0.0.1:8080/` in your web browser.
2.  Paste the news article text into the textarea.
3.  Click "Analyze News".
4.  The prediction (REAL or FAKE) along with a confidence score will be displayed.

## How It Works

1.  The user submits news text through the Django web interface.
2.  The text is preprocessed (lowercase, remove punctuation, stopwords, etc.) using the same steps applied during model training.
3.  The preprocessed text is transformed into a numerical vector using the pre-trained TF-IDF vectorizer.
4.  This vector is fed into the pre-trained Logistic Regression model, which outputs a prediction (0 for FAKE, 1 for REAL) and probabilities.
5.  The result is displayed back to the user.

## Troubleshooting Git LFS

*   If `news.csv` is very small (e.g., a few hundred bytes) after cloning, Git LFS might not have pulled the actual file content. Ensure Git LFS is installed and run:
    ```bash
    git lfs pull
    ```
    in the repository directory.

## Future Improvements

*   Incorporate more advanced NLP models (e.g., Transformers like BERT, RoBERTa).
*   Expand the training dataset for better generalization.
*   Add user authentication and a history of predictions.
*   Deploy the application to a cloud platform (e.g., Heroku, AWS, Google Cloud).
*   Improve UI/UX.

