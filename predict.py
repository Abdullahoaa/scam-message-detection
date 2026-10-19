import os
import joblib
import numpy as np
import re
from scipy.sparse import hstack
from text_preprocessor import AdvancedTextPreprocessor

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

class ScamPredictor:
    def __init__(self):
        self.logistic_model = joblib.load(os.path.join(MODEL_DIR, 'scam_detection_logistic_regression.pkl'))
        self.rf_model = joblib.load(os.path.join(MODEL_DIR, 'scam_detection_random_forest.pkl'))
        self.tfidf_scam = joblib.load(os.path.join(MODEL_DIR, 'tfidf_scam.pkl'))
        self.tfidf_response = joblib.load(os.path.join(MODEL_DIR, 'tfidf_response.pkl'))
        self.preprocessor = AdvancedTextPreprocessor()
        self.safe_numerical_features = joblib.load(os.path.join(MODEL_DIR, 'numerical_features.pkl'))

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s\.!?,]', '', text)
        return ' '.join(text.split())

    def _encode_response_strategy(self, response_type):
        response_strategy_map = {
            'questioning': 1,
            'delaying': 2,
            'refusing': 3,
            'counter_attack': 4,
            'verification': 5,
            'expert_mention': 6
        }
        return response_strategy_map.get(response_type, 0)

    def _encode_risk_level(self, value):
        if isinstance(value, str):
            level_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            return level_map.get(value.lower(), 0)
        if isinstance(value, (int, float)):
            if value <= 5:
                return 1
            if value <= 7:
                return 2
            if value <= 9:
                return 3
            return 4
        return 0

    def _detect_indicators(self, message, response):
        scam_text = self._clean_text(message)
        response_text = self._clean_text(response)

        indicators = {
            'urgency_keywords': int(bool(re.search(r'\b(urgent|urgently|immediately|asap|now|hurry|deadlines?|limited time|24 hours|today|expires?|expiring)\b', scam_text))),
            'fake_fee_mentions': int(bool(re.search(r'\b(fees?|charges?|payments?|invoices?|processing|costs?|deposits?|service fee|security fee|pay)\b', scam_text))),
            'emotional_manipulation': int(bool(re.search(r'\b(fear|panic|lose|losing|love|rewards?|gifts?|help me|please)\b', scam_text))),
            'isolation_attempts': int(bool(re.search(r'\b(private|confidential|secret|only you|do not tell|dont tell|tell anyone|no one else)\b', scam_text))),
            'authority_claims': int(bool(re.search(r'\b(official|admin|administrator|manager|bank|fbi|irs|police|security|government|law enforcement)\b', scam_text))),
            'threat_indicators': int(bool(re.search(r'\b(block|blocked|suspend|suspended|suspension|fines?|fined|penalty|penalties|lock|locked|terminate|terminated|termination|shut down|disabled|arrest|arrested)\b', scam_text)))
        }

        indicators['total_scam_indicators'] = sum(indicators.values())
        indicators['contains_verification'] = int(bool(re.search(r"\b(verify|verification|confirm|identity|id|authenticate)\b", scam_text)))
        indicators['contains_refusal'] = int(bool(re.search(r"\b(no|not|never|dont|can't|cannot|won't|will not|refuse|decline)\b", response_text)))
        indicators['contains_expertise'] = int(bool(re.search(r"\b(cybersecurity|security expert|expert|professional|specialist|researcher|analyst)\b", response_text)))

        response_strength = 3
        if indicators['contains_refusal']:
            response_strength += 1
        if indicators['contains_expertise']:
            response_strength += 1
        if len(response_text.split()) > 15:
            response_strength += 1
        indicators['response_strength'] = min(max(response_strength, 3), 5)

        return indicators

    def _estimate_risk_score(self, indicators):
        score = 4 + indicators['total_scam_indicators']
        if indicators['authority_claims']:
            score += 1
        if indicators['threat_indicators']:
            score += 1
        return min(score, 10)

    def _risk_label(self, score):
        if score <= 5:
            return 'LOW'
        if score <= 7:
            return 'MEDIUM'
        if score <= 9:
            return 'HIGH'
        return 'CRITICAL'

    def _estimate_effectiveness_score(self, response_text, indicators):
        score = 3
        if indicators['contains_refusal']:
            score += 1
        if indicators['contains_expertise']:
            score += 1
        if len(response_text.split()) > 20:
            score += 1
        return min(score, 5)

    def _encode_response_type(self, response):
        response = response.lower()
        if re.search(r"\b(why|what|how|when|where|who)\b", response):
            return 1
        if re.search(r"\b(later|delay|wait|not now|postpone)\b", response):
            return 2
        if re.search(r"\b(no|not|never|cannot|cant|won't|will not|refuse|decline)\b", response):
            return 3
        if re.search(r"\b(know|expert|cybersecurity|security|professional|specialist)\b", response):
            return 6
        if re.search(r"\b(verify|verification|confirm|authenticate)\b", response):
            return 5
        if re.search(r"\b(attack|fight|challenge|counter)\b", response):
            return 4
        return 0

    def extract_features(self, scam_message, user_response, metadata=None):
        scam_processed = self.preprocessor.advanced_preprocess(scam_message)
        response_processed = self.preprocessor.advanced_preprocess(user_response)

        scam_text_analysis = self.preprocessor.analyze_text_complexity(scam_message)
        response_text_analysis = self.preprocessor.analyze_text_complexity(user_response)

        scam_tfidf = self.tfidf_scam.transform([scam_processed])
        response_tfidf = self.tfidf_response.transform([response_processed])

        indicators = self._detect_indicators(scam_message, user_response)
        indicators['risk_score'] = self._estimate_risk_score(indicators)
        indicators['effectiveness_score'] = self._estimate_effectiveness_score(user_response, indicators)
        indicators['response_strategy_encoded'] = self._encode_response_type(user_response)
        indicators['risk_level_encoded'] = self._encode_risk_level(indicators['risk_score'])

        # Text-complexity stats were used as features in training (scam_*/response_*),
        # so they must be supplied at inference too, otherwise they silently default to 0.
        model_features = dict(indicators)
        for prefix, stats in (('scam_', scam_text_analysis), ('response_', response_text_analysis)):
            for key, value in stats.items():
                model_features[prefix + key] = value

        if metadata:
            overrides = {k: metadata[k] for k in metadata if k in model_features}
            indicators.update({k: v for k, v in overrides.items() if k in indicators})
            model_features.update(overrides)

        numerical_array = np.array([model_features.get(feature, 0) for feature in self.safe_numerical_features], dtype=float).reshape(1, -1)
        combined_features = hstack([scam_tfidf, response_tfidf, numerical_array])

        return {
            'combined_features': combined_features,
            'scam_message': scam_message,
            'user_response': user_response,
            'analysis': {
                'scam': scam_text_analysis,
                'response': response_text_analysis,
                'metadata': indicators
            }
        }

    def predict(self, scam_message, user_response='', metadata=None):
        user_response = user_response or ''
        data = self.extract_features(scam_message, user_response, metadata)
        metadata_out = data['analysis']['metadata']

        risk_score = metadata_out['risk_score']
        data['risk_assessment'] = {
            'label': self._risk_label(risk_score),
            'risk_score': risk_score,
            'confidence_percent': round(risk_score / 10 * 100, 1),
            'total_indicators': metadata_out['total_scam_indicators']
        }

        has_response = bool(user_response.strip())
        if has_response:
            combined_features = data['combined_features']

            lr_proba = self.logistic_model.predict_proba(combined_features)[0][1]
            rf_proba = self.rf_model.predict_proba(combined_features)[0][1]

            lr_pred = int(lr_proba > 0.5)
            rf_pred = int(rf_proba > 0.5)
            ensemble_proba = float((lr_proba + rf_proba) / 2)
            ensemble_pred = int(ensemble_proba > 0.5)

            data['response_effectiveness'] = {
                'logistic_regression': {
                    'label': 'EFFECTIVE' if lr_pred == 1 else 'INEFFECTIVE',
                    'score': float(lr_proba)
                },
                'random_forest': {
                    'label': 'EFFECTIVE' if rf_pred == 1 else 'INEFFECTIVE',
                    'score': float(rf_proba)
                },
                'ensemble': {
                    'label': 'EFFECTIVE' if ensemble_pred == 1 else 'INEFFECTIVE',
                    'score': ensemble_proba
                }
            }
        else:
            data['response_effectiveness'] = None

        return data
