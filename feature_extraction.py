# feature_extraction.pyTruncatedSVD
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

def extract_features(data):
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(data)
    
    lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)
    
    explained_variance = lsa[0].explained_variance_ratio_.sum()
    
    return X_lsa, explained_variance
