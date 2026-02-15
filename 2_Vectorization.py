import pandas as pd
import pickle
import os
import sys
import multiprocessing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# ==========================================
# KONFIGURATION
# ==========================================
INPUT_FILE = "data/corpus_cleaned.pkl"
OUTPUT_TFIDF = "data/tfidf_data.pkl"
OUTPUT_W2V = "data/w2v_model.model"

# ==========================================
# TF-IDF FUNKTION
# ==========================================
def run_tfidf(df):
    """
    Erstellt eine Document-Term-Matrix basierend auf gewichteten Worthäufigkeiten.
    Ziel ist es, Wörter finden, die für ein Dokument spezifisch sind, höher zu gewichten
    während Füllwörter, die überall vorkommen, abgewertet werden.
    """
    print("\n" + "="*50)
    print("   METHODE A: Statistische Vektorisierung (TF-IDF)")
    print("="*50)

    # 1. Konfiguration des Vectorizers
    print("1. Konfiguriere TF-IDF Parameter...")
    # max_df=0.90: Ignoriert Wörter, die in mehr als 90% der Dokumente vorkommen ("the", "consumer", "complaint").
    # min_df=2:    Ignoriert Wörter, die in weniger als 2 Dokumenten vorkommen (Tippfehler, Ausreißer).
    # max_features=1000: Begrenzt die Matrix auf die 1000 stärksten Wörter (optimiert Rechenzeit).
    vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
    
    # 2. Berechnung der Matrix
    print("2. Berechne Matrix...")
    # fit_transform lernt das Vokabular und transformiert den Text in Zahlen
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    # Ausgabe der Form: (Anzahl Dokumente, Anzahl Wörter)
    print(f"   Ergebnis-Form: {tfidf_matrix.shape} (Dokumente x Features)")
    
    # 3. Validierung
    # Wir schauen uns beispielhaft an, welche Wörter im ersten Dokument wichtig sind.
    if len(df) > 0:
        feature_names = vectorizer.get_feature_names_out()
        # Holt den Vektor des ersten Dokuments
        first_vec = tfidf_matrix[0].T.todense()
        # Erstellt einen kleinen DataFrame zur Anzeige
        df_check = pd.DataFrame(first_vec, index=feature_names, columns=["score"])
        # Sortiert nach Score (Wichtigstes oben)
        top_words = df_check.sort_values(by="score", ascending=False).head(5)
        
        print("\n--- Validierung: Wichtigste Keywords in Dokument 1 ---")
        # Zeige nur Wörter an, die auch wirklich im Dokument vorkommen (Score > 0)
        print(top_words[top_words['score'] > 0])

    # 4. Speichern für spätere Nutzung
    # Wir speichern sowohl die Matrix als auch den Vectorizer
    with open(OUTPUT_TFIDF, "wb") as f:
        pickle.dump((tfidf_matrix, vectorizer), f)
    print(f"TF-IDF Matrix erfolgreich gespeichert in '{OUTPUT_TFIDF}'")

# ==========================================
# WORD2VEC FUNKTION
# ==========================================
def run_word2vec(df):
    """
    Trainiert ein Word Embedding Modell.
    Ziel ist es, Wörter in Vektoren umwandeln, sodass Wörter mit ähnlicher Bedeutung
    im mathematischen Raum nah beieinander liegen.
    """
    print("\n" + "="*50)
    print("   METHODE B: Semantische Vektorisierung (Word2Vec)")
    print("="*50)

    # 1. Vorbereitung (Tokenisierung)
    # Gensim benötigt eine Liste von Listen: [['wort1', 'wort2'], ['wort3', 'wort4']]
    print("1. Bereite Sätze für das neuronale Netz vor...")
    tokenized_sentences = [text.split() for text in df['clean_text']]
    
    # 2. Training des Modells
    # Nutzt alle verfügbaren CPU-Kerne minus 1, um das System nicht einzufrieren
    cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"2. Starte Training mit {cores} CPU-Kernen...")
    
    model = Word2Vec(
        sentences=tokenized_sentences, 
        vector_size=100, # Dimension: Jedes Wort wird durch 100 Zahlen repräsentiert
        window=5,        # Kontext: Das Modell schaut 5 Wörter nach links und rechts
        min_count=2,     # Rauschen: Wörter, die nur 1x vorkommen, werden ignoriert
        workers=cores,   # Parallelisierung
        seed=42          # Reproduzierbarkeit
    )
    
    # 3. Validierung (Semantik-Check)
    # Testet qualitativ, ob das Modell "verstanden" hat, was die Wörter bedeuten.
    print("\n--- Validierung: Semantische Ähnlichkeiten ---")
    check_terms = ['money', 'credit', 'bank', 'scam']
    
    for term in check_terms:
        # Kontrolliert, ob das Wort im Vokabular ist 
        if term in model.wv:
            # most_similar berechnet die Kosinus-Ähnlichkeit im Vektorraum
            similar = model.wv.most_similar(term, topn=3)
            words = [w[0] for w in similar]
            print(f"   Kontext zu '{term}': {words}")
        else:
            print(f"   Begriff '{term}' wurde weggefiltert (zu selten).")

    # 4. Speichern des Modells
    model.save(OUTPUT_W2V)
    print(f"Word2Vec Modell erfolgreich gespeichert in '{OUTPUT_W2V}'")

# ==========================================
# MAIN
# ==========================================
def main():
    print("==========================================")
    print("   PHASE 2: Vektorisierung (Vergleich)")
    print("==========================================")

    # 1. Eingabedatei prüfen
    if not os.path.exists(INPUT_FILE):
        print(f"FEHLER: Datei '{INPUT_FILE}' fehlt.")
        print("Bitte führe zuerst '1_Data_Pipeline.py' aus.")
        return

    # 2. Daten laden (nur einmal für beide Methoden!)
    print("Lade bereinigte Daten...")
    df = pd.read_pickle(INPUT_FILE)
    
    # Sicherheitscheck: Leere Texte entfernen, um Fehler bei der Berechnung zu vermeiden
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    print(f"Daten geladen: {len(df)} Dokumente bereit zur Analyse.")

    # 3. Methode A ausführen (Statistik)
    run_tfidf(df)

    # 4. Methode B ausführen (Semantik)
    run_word2vec(df)

    # 5. Abschlussbericht
    print("\n" + "="*50)
    print("   ABSCHLUSS PHASE 2")
    print("="*50)
    print("Beide Vektorisierungsarten wurden erfolgreich durchgeführt.")
    print("-> TF-IDF Matrix liegt in 'data/tfidf_data.pkl'")
    print("-> Word2Vec Modell liegt in 'data/w2v_model.model'")

if __name__ == "__main__":
    main()