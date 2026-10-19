import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from text_preprocessor import AdvancedTextPreprocessor

import sys

# Path to the training CSV: pass it as the first argument or set SCAM_DATASET_PATH.
DATA_PATH = (sys.argv[1] if len(sys.argv) > 1 else
             os.environ.get('SCAM_DATASET_PATH', 'unified_scam_detection_dataset.csv'))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(path):
    return pd.read_csv(path)


def build_feature_pipeline(df, preprocessor):
    df = df.copy()
    df['scam_processed'] = df['scammer_message'].astype(str).apply(preprocessor.advanced_preprocess)
    df['response_processed'] = df['user_response'].astype(str).apply(preprocessor.advanced_preprocess)

    tfidf_scam = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5,
        max_df=0.7,
        sublinear_tf=True
    )

    tfidf_response = TfidfVectorizer(
        max_features=400,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=5,
        max_df=0.8,
        sublinear_tf=True
    )

    scam_tfidf = tfidf_scam.fit_transform(df['scam_processed'])
    response_tfidf = tfidf_response.fit_transform(df['response_processed'])

    response_strategy_map = {
        'questioning': 1,
        'delaying': 2,
        'refusing': 3,
        'counter_attack': 4,
        'verification': 5,
        'expert_mention': 6
    }

    risk_level_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    df['response_strategy_encoded'] = df['response_type'].map(response_strategy_map).fillna(0).astype(int)
    df['risk_level_encoded'] = df['risk_level'].map(risk_level_map).fillna(0).astype(int)

    numerical_features = [
        'risk_score', 'effectiveness_score', 'urgency_keywords', 'fake_fee_mentions',
        'emotional_manipulation', 'isolation_attempts', 'authority_claims',
        'threat_indicators', 'total_scam_indicators', 'contains_verification',
        'contains_refusal', 'contains_expertise', 'response_strength',
        'response_strategy_encoded', 'risk_level_encoded',
        'scam_word_count', 'scam_sentence_count', 'scam_avg_sentence_length',
        'scam_lexical_diversity', 'scam_polarity', 'scam_subjectivity',
        'response_word_count', 'response_lexical_diversity', 'response_polarity',
        'response_subjectivity'
    ]

    scam_analysis = df['scammer_message'].astype(str).apply(preprocessor.analyze_text_complexity).apply(pd.Series)
    response_analysis = df['user_response'].astype(str).apply(preprocessor.analyze_text_complexity).apply(pd.Series)
    df = pd.concat([df, scam_analysis.add_prefix('scam_'), response_analysis.add_prefix('response_')], axis=1)

    df[numerical_features] = df[numerical_features].fillna(0)
    X_numerical = df[numerical_features].values
    X_combined = hstack([scam_tfidf, response_tfidf, X_numerical])

    return X_combined, df, tfidf_scam, tfidf_response, numerical_features


def train_and_serialize():
    df = load_dataset(DATA_PATH)
    preprocessor = AdvancedTextPreprocessor()
    X_combined, df, tfidf_scam, tfidf_response, numerical_features = build_feature_pipeline(df, preprocessor)
    y = df['user_success']
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

    lr_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1)

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    joblib.dump(lr_model, os.path.join(MODEL_DIR, 'scam_detection_logistic_regression.pkl'))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'scam_detection_random_forest.pkl'))
    joblib.dump(tfidf_scam, os.path.join(MODEL_DIR, 'tfidf_scam.pkl'))
    joblib.dump(tfidf_response, os.path.join(MODEL_DIR, 'tfidf_response.pkl'))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'text_preprocessor.pkl'))
    joblib.dump(numerical_features, os.path.join(MODEL_DIR, 'numerical_features.pkl'))

    print('Models and preprocessing objects saved to', MODEL_DIR)


if __name__ == '__main__':
    train_and_serialize()
