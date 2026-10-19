# Scam Message Detection

A Flask web app that scans a suspicious text message or email for common scam tactics and gives a clear risk assessment. If you have already replied to the message, it also rates how well your reply resisted the scam using a machine-learning ensemble (Logistic Regression + Random Forest).

## Features

- **Scam risk scoring**: detects six cybersecurity indicators (urgency, threats, authority claims, fake fees, emotional manipulation, isolation attempts) and rates the message LOW / MEDIUM / HIGH / CRITICAL.
- **Response effectiveness** (optional): an ensemble of Logistic Regression and Random Forest models, trained on scam conversation data, predicts whether your reply was effective.
- **Tailored recommendations** based on which tactics were detected.
- **Private**: everything runs locally, and messages are processed in memory and never stored.

## Tech stack

Python, Flask, scikit-learn, NLTK, TextBlob, SciPy, pandas, plus HTML/CSS/JavaScript for the UI.

## Project structure

```
├── app.py                  # Flask server: routes and recommendations
├── predict.py              # ScamPredictor: indicator detection, features, model inference
├── text_preprocessor.py    # Text cleaning, lemmatization, and text-complexity stats
├── train_model.py          # Training script that produces the .pkl files
├── *.pkl                   # Trained models, TF-IDF vectorizers, feature list
├── templates/index.html    # Web UI
└── static/                 # styles.css, app.js
```

## Getting started

```bash
git clone https://github.com/Abdullahoaa/scam-message-detection.git
cd scam-message-detection
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 in your browser. On first run, NLTK downloads its data (punkt, stopwords, wordnet), so the first start needs an internet connection.

Optional environment variables:
- `PORT`: the port to serve on (default `5000`)
- `FLASK_DEBUG=1`: turns on Flask debug mode for development

## API

`POST /analyze` with a JSON body:

```json
{
  "message": "URGENT: your account will be suspended unless you pay a fee now.",
  "user_response": "No, I will verify with my bank directly."
}
```

The response contains `risk_assessment`, `response_effectiveness` (`null` when no reply is given), `analysis`, and `recommendations`.

## Retraining the models

The training dataset isn't included in this repo. Pass its path to retrain:

```bash
python train_model.py path/to/unified_scam_detection_dataset.csv
# or: set SCAM_DATASET_PATH and run python train_model.py
```

The CSV needs the columns `scammer_message`, `user_response`, `response_type`, `risk_level`, `user_success`, and the numeric indicator columns listed in `train_model.py`.

> The included `.pkl` files were saved with scikit-learn 1.8.0, which is why `requirements.txt` pins it. If you upgrade scikit-learn, retrain the models.

## Limitations

- The scam risk score comes from keyword/indicator rules, not from the ML model, so cleverly worded scams can slip through.
- The response-effectiveness models sometimes disagree, and they can underrate good replies that aren't like the training data. Treat the score as guidance, not a verdict.
- Supports English only.

## Author

Abdullah Omar AL-Abdullatif, Computer Science, College of IT, King Faisal University
[GitHub](https://github.com/Abdullahoaa) · [LinkedIn](https://www.linkedin.com/in/abdullah-al-abdullatif-736b16380/)
