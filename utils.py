import joblib
import re
from nltk.tokenize import WordPunctTokenizer
# load the model and vectorizer
stp_ = joblib.load("add_stopword.joblib")
model = joblib.load('model_svc_20221121.joblib')
vectorizer = joblib.load('tfidf_svc_20221121.joblib')

def text_cleaner(text, stop_word=True, add_stp = []):
    
    tok = WordPunctTokenizer()

    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    pat3 = '([^0-9A-Za-z \t])|(\w+:\/\/\S+)'
    combined_pat = r'|'.join((pat1, pat2, pat3))

    stripped = re.sub(combined_pat, '', text)

    try:
        clean = stripped.decode("utf-8").replace(u"\ufffd", "?")
    except:
        clean = stripped

    lower_case = clean.lower()

    words = tok.tokenize(lower_case)

    if stop_word:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
        
        sw = StopWordRemoverFactory().get_stop_words()
        edit_sword = sw+add_stp

        filtered_words = [item for item in words if item not in edit_sword]

        return (" ".join(filtered_words)).strip()
    else:
        return (" ".join(words)).strip()


def predict_text(text):
    # make a prediction based on the transformed text
    txt = text_cleaner(text, stop_word=True, add_stp = stp_)
    pred = model.predict(vectorizer.transform([txt]))[0]
    return pred
