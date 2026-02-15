import pandas as pd
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore
import os
import time

# ==========================================
# KONFIGURATION
# ==========================================
# Input und Output
INPUT_FILE = "data/corpus_cleaned.pkl" # Datensatz wird geladen
PLOT_FILE  = "data/coherence_plot.png"  # Name der erstellten Grafik

# Suchbereich für die Themenanzahl
K_START = 4    # Start bei x Themen
K_LIMIT = 8   # Limit bei x Themen
K_STEP  = 1    # Jeden x Schritt testen (1,2,3 oder 2,4,6)

# Performance & Sampling
SAMPLE_FRAC = 0.80  # Prozentsatz verarbeiteter Daten (0.20 = 20% - 1.00 = 100%)
WORKERS     = 14     # Zahl der verwendeten CPU-Kerne, um den Prozess zu beschleunigen
PASSES      = 2     # Anzahl der Durchläuft um die Präzision zu verbessern

# ==========================================
# FUNKTION: Berechnung der Scores
# ==========================================
def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
    """
    Berechnet den Coherence Score (C_v) für verschiedene Anzahlen von Themen (K).
    Rückgabe:
    - x_values         : Liste der getesteten K-Werte
    - coherence_values : Liste der zugehörigen Scores
    """
    coherence_values = []
    x_values = range(start, limit, step)

    print(f"\nStarte Hyperparameter-Optimierung (K={start} bis {limit}, Schritt={step}) ---")

    for num_topics in x_values:
        start_time = time.time()
        print(f"   > Teste K={num_topics}...", end=" ", flush=True)
        
        # 1. Modelltraining (Multicore)
        # Nutzt die angegebene Zahl an CPU-Kernen, um das Training zu beschleunigen
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics, 
            random_state=42,
            passes=PASSES,      
            workers=WORKERS,    
            chunksize=2000      
        )
        
        # 2. Score-Berechnung
        # Nutzt die Coherence Value (c_v), um die Top-Wörter eines Themas zu messen.
        coherencemodel = CoherenceModel(
            model=model, 
            texts=texts, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        score = coherencemodel.get_coherence()
        coherence_values.append(score)
        
        duration = time.time() - start_time
        print(f"Score: {score:.4f} (Dauer: {duration:.1f}s)")

    return x_values, coherence_values

# ==========================================
# MAIN
# ==========================================
def main():
    print("==========================================")
    print("     PHASE 3: Coherence Analysis (Sampled)")
    print("==========================================")
    
    # Daten prüfen
    if not os.path.exists(INPUT_FILE):
        print(f"[FEHLER] Datei '{INPUT_FILE}' nicht gefunden.")
        return

    print("Lade Daten...")
    df = pd.read_pickle(INPUT_FILE)
    initial_len = len(df)
    
    # ===== SAMPLING =====
    # Zieht zufällig einen oben angegebenen Prozentsatz der Daten.
    # Das verhindert Abstürze bei begrenztem RAM und ermöglicht Testläufe.
    # random_state=42 garantiert die Reproduzierbarkeit - es wird immer derselbe Teil der Stichproben verarbeitet.
    df = df.sample(frac=SAMPLE_FRAC, random_state=42)
    
    # Text bereinigen (leere Zeilen raus)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    
    print(f"   Original: {initial_len} Dokumente")
    print(f"   Analyse auf Sample: {len(df)} Dokumente (reicht für K-Bestimmung)")
    
    # Tokenisierung für Gensim (Wortlisten statt Strings)
    print("Tokenisierung...")
    tokenized_text = [text.split() for text in df['clean_text']]

    # 2. Dictionary und Corpus erstellen
    print("Dictionary & Corpus...")
    dictionary = corpora.Dictionary(tokenized_text)
    # no_below=3: Filtert wörter, die weniger als 3 mal im Corpus vorkommen (Tippfehler, Ausreißer)
    # no_above=0.9: Filtert Wörter, die in über 90% der Dokumente vorkommen (Stopwörter wie "complaint")
    dictionary.filter_extremes(no_below=3, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]

    # Hauptfunktion starten
    x, y = compute_coherence_values(
        dictionary=dictionary, 
        corpus=corpus, 
        texts=tokenized_text, 
        start=K_START, 
        limit=K_LIMIT, 
        step=K_STEP
    )

    # Ergebnis ausgeben - Text und Grafik
    if y:
        best_idx = y.index(max(y))
        best_k = x[best_idx]
        print(f"\nOptimum: K={best_k} (Score: {max(y):.4f})")
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', color='b')
        plt.plot(best_k, max(y), marker='o', color='r', markersize=10)
        plt.title(f"Optimierung (Sample {SAMPLE_FRAC*100}%, Multicore)")
        plt.xlabel("Anzahl der Themen (K)")
        plt.ylabel("Coherence Score (C_v)")
        plt.grid(True)
        plt.savefig(PLOT_FILE)
        print(f"Grafik gespeichert: {PLOT_FILE}")
        print("-" * 30)
        print(f"Empfehlung: K={best_k}.")
        print("-" * 30)

if __name__ == "__main__":
    main()