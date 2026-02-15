import pandas as pd
import os
import sys
import gc
import multiprocessing
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import gensim
from gensim import corpora
from gensim.models import LdaMulticore

# ==========================================
# KONFIGURATION
# ==========================================
INPUT_FILE = "data/corpus_cleaned.pkl"
OUTPUT_FILE = "data/topics_output.txt"

# Variablen zur Anpassung von Umfang und Systemauslastung
SAMPLE_FRAC = 0.60 # Sampling-Rate (0.60 = 60% der Daten)
NUM_TOPICS  = 5 # Empfohlene Themenzahl K
WORKERS = 4 # Zahl der verwendeten CPU-Kerne

def save_output(text):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def main():
    start_global = time.time()
    print("==========================================")
    print(f"   TOPIC MODELING (RAM-Optimized)")
    print("==========================================")

    # 1. Output Datei resetten
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"REPORT START: {time.ctime()}\nSample-Rate: {SAMPLE_FRAC}\n\n")

    if not os.path.exists(INPUT_FILE):
        print("Fehler: Input-Datei fehlt.")
        return

    # ==========================================
    # SCHRITT A: DATEN LADEN UND SAMPELN
    # ==========================================
    print("1. Lade DataFrame...")
    df = pd.read_pickle(INPUT_FILE)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    
    if SAMPLE_FRAC < 1.0:
        print(f"   Sampling {int(SAMPLE_FRAC*100)}%...")
        df = df.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)
        gc.collect() # RAM aufräumen

    print(f"   Verarbeite {len(df)} Dokumente.")

    # ==========================================
    # SCHRITT B: NMF (Scikit-Learn)
    # NMF braucht den Text als Strings,
    # daher wird es ausgeführt, solange 'df' noch existiert.
    # ==========================================
    print("\n2. Starte NMF (Matrix Factorization)...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000)
    tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    nmf_model = NMF(n_components=NUM_TOPICS, random_state=42, max_iter=500)
    nmf_model.fit(tfidf)
    
    out_nmf = "\n=== NMF ERGEBNISSE ===\n"
    for idx, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        line = f"Thema {idx+1}: {', '.join(top_words)}"
        print(f"   {line}")
        out_nmf += line + "\n"
    save_output(out_nmf)

    # RAM sparen: TF-IDF wird nicht mehr bnötigt
    del tfidf, tfidf_vectorizer, nmf_model
    gc.collect()

    # ==========================================
    # SCHRITT C: VORBEREITUNG LDA
    # ==========================================
    print("\n3. Vorbereitung für LDA...")
    
    # Tokenisierung
    # Das erstellt eine riesige Liste im RAM.
    texts = [text.split() for text in df['clean_text']]
    
    # Den DataFrame löschen - das gibt RAM frei
    print("   -> Lösche DataFrame aus RAM...")
    del df
    gc.collect()

    # Dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    
    # Corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Hier werden auch die Texte gelöscht
    # LDA braucht nur den Corpus (Zahlen), keine Wörter.
    print("   -> Lösche Token-Liste aus RAM...")
    del texts
    gc.collect()
    
    print("   RAM ist bereinigt. Starte Multicore-Training...")

    # ==========================================
    # SCHRITT D: LDA TRAINING
    # ==========================================
    t_start = time.time()
    
    # Initialisierung des parallelen LDA-Modells
    lda_model = LdaMulticore(
        corpus=corpus,          # Der Datensatz als Bag-of-Words
        id2word=dictionary,     # Mapping von Wort-IDs zurück zu echten Wörtern
        num_topics=NUM_TOPICS,  # Die ermittelte optimale Anzahl an Themen
        workers=WORKERS,        # Anzahl der CPU-Kerne
        passes=10,              # Anzahl der kompletten Durchläufe
        chunksize=2000,         # Anzahl der Dokumente, die gleichzeitig in den RAM geladen
        random_state=42
    )
    
    duration = (time.time() - t_start) / 60
    print(f"LDA Fertig in {duration:.1f} Minuten.")

    out_lda = "\n=== LDA ERGEBNISSE ===\n"
    for idx, topic in lda_model.print_topics(-1):
        clean = topic.replace('"', '').replace(' + ', ', ')
        line = f"Thema {idx+1}: {clean}"
        print(f"   {line}")
        out_lda += line + "\n"
    save_output(out_lda)

    print(f"\nGesamtdauer: {(time.time() - start_global)/60:.1f} Minuten.")
    print(f"Ergebnisse gespeichert in '{OUTPUT_FILE}'")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()