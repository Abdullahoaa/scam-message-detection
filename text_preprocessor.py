import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob


def ensure_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class AdvancedTextPreprocessor:
    def __init__(self):
        ensure_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s\.\?!,]', '', text)
        return ' '.join(text.split())

    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def analyze_text_complexity(self, text):
        if not isinstance(text, str):
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'lexical_diversity': 0,
                'polarity': 0,
                'subjectivity': 0
            }
        words = word_tokenize(text)
        sentences = [s for s in text.split('.') if s.strip()]
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        avg_sentence_length = word_count / sentence_count
        unique_words = len(set(words))
        lexical_diversity = unique_words / max(1, word_count)
        sentiment = TextBlob(text).sentiment
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'lexical_diversity': lexical_diversity,
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity
        }

    def advanced_preprocess(self, text):
        cleaned_text = self.clean_text(text)
        return self.tokenize_and_lemmatize(cleaned_text)
