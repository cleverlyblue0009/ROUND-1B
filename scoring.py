# scoring.py
import re
import math
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {
    "a","an","the","of","to","and","in","on","for","with","by","is","are","be",
    "was","were","as","at","from","or","that","this","it","its","into","about",
    "your","their","his","her","our","we","you","i"
}

WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def tokenize(text:str)->List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]

def keyword_set(text:str)->set:
    return {w for w in tokenize(text) if w not in STOPWORDS and len(w)>2}

def build_query(persona:str, job:str)->str:
    # simple concat
    return f"{persona} {job}"

def build_keywords(persona:str, job:str)->set:
    ks=keyword_set(persona) | keyword_set(job)
    return ks

def tfidf_scores(query:str, sections:List[str])->np.ndarray:
    # fit joint vectorizer (no external resources)
    vec=TfidfVectorizer(lowercase=True, stop_words=list(STOPWORDS))
    mat=vec.fit_transform([query]+sections)  # row0=query
    sims=cosine_similarity(mat[0], mat[1:]).flatten()
    return sims

def heading_overlap_score(heading:str, kw:set)->float:
    hkw=keyword_set(heading)
    if not hkw: return 0.0
    overlap=len(hkw & kw)/len(hkw)
    return overlap

def domain_boost_score(text:str, kw:set)->float:
    toks=keyword_set(text)
    if not toks: return 0.0
    overlap=len(toks & kw)
    # log scaled
    return math.log1p(overlap)/5.0  # ~0-0.4

def combined_scores(query:str, headings:List[str], texts:List[str], kw:set,
                    w_tfidf=0.6, w_head=0.25, w_domain=0.15)->np.ndarray:
    tfidf=tfidf_scores(query,texts)
    head=np.array([heading_overlap_score(h,kw) for h in headings])
    domain=np.array([domain_boost_score(t,kw) for t in texts])
    score = w_tfidf*tfidf + w_head*head + w_domain*domain
    return score
