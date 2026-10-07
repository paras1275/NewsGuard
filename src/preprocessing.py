import re
import string
# Safe NLTK import with fallbacks
try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore


    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

    STOP_WORDS = set(stopwords.words('english'))
    LEMMATIZER = WordNetLemmatizer()
except ImportError:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    }
    class SimpleLemmatizer:
        def lemmatize(self, word):
            return word.rstrip('s') if len(word) > 3 and word.endswith('s') else word
    LEMMATIZER = SimpleLemmatizer()


# Attempt spacy load safely with fallback
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        nlp = None
except ImportError:
    nlp = None


def clean_text(text: str) -> str:
    """
    Cleans raw text:
    - Lowercasing
    - Removes URLs
    - Removes HTML tags
    - Removes special characters and punctuation
    - Normalizes numbers
    - Strips unnecessary whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https\S+', ' ', text)

    # Replace numbers with a space or normalized token
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation & special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text: str) -> list:
    """Tokenizes cleaned text into a list of words."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def lemmatize_text(tokens: list) -> list:
    """
    Lemmatizes tokens using SpaCy if available, otherwise NLTK WordNetLemmatizer.
    Filters out English stopwords and short tokens (< 2 chars).
    """
    if not tokens:
        return []

    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    if nlp is not None:
        doc = nlp(" ".join(tokens))
        lemmas = [token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 1]
        return lemmas
    else:
        # Fallback to NLTK WordNet lemmatizer
        lemmas = [LEMMATIZER.lemmatize(word) for word in tokens]
        return lemmas


def preprocess_article(text: str) -> str:
    """
    Complete end-to-end NLP preprocessing pipeline for a news article string.
    Returns clean, lemmatized text joined by single space.
    """
    tokens = tokenize_text(text)
    lemmas = lemmatize_text(tokens)
    return " ".join(lemmas)


if __name__ == "__main__":
    sample = "<h3>SHOCKING NEWS!</h3> Check http://example.com for 100% true facts! Doctors are furious!!!"
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))
    print("Preprocessed:", preprocess_article(sample))
