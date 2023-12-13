# preprocessing.py
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def remove_punctuation(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\s+', ' ', text)
    return text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def preprocess_text(text, lang='english'):
    text = remove_punctuation(text.lower())
    words = word_tokenize(text)
    stop_words = set(stopwords.words(lang))
    words = [word for word in words if word not in stop_words]
    tagged = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    result = [lemmatizer.lemmatize(word[0], pos=get_wordnet_pos(word[1]) or wordnet.NOUN) for word in tagged]
    result = ' '.join(result)
    return result
