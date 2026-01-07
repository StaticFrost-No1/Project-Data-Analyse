import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Konfiguration
INPUT_FILE = "data/corpus_cleaned.pkl" # Oder "corpus_cleaned.pkl", falls ohne Ordner
# Wir nutzen nur die Texte, die nicht leer sind
df = pd.read_pickle(INPUT_FILE)
print(f"Daten geladen: {len(df)} Zeilen.")

# Sicherstellen, dass keine leeren Strings dabei sind (kann durch Cleaning passieren)
df = df[df['clean_text'].str.len() > 0].copy()

# ==========================================
# ANSATZ 1: TF-IDF (Statistisch)
# ==========================================
print("\n--- Start TF-IDF Vektorisierung ---")

# Initialisierung: Wir ignorieren Wörter, die in mehr als 90% der Doks vorkommen (max_df)
# und Wörter, die in weniger als 2 Doks vorkommen (min_df)
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)

# Matrix erstellen
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])

print(f"TF-IDF Matrix Form: {tfidf_matrix.shape}")
print("(Anzahl Dokumente, Anzahl Wörter im Vokabular)")

# Beispiel: Die Top-Wörter im ersten Dokument anzeigen
feature_names = tfidf_vectorizer.get_feature_names_out()
first_doc_vector = tfidf_matrix[0]
df_tfidf = pd.DataFrame(first_doc_vector.T.todense(), index=feature_names, columns=["tfidf"])
print("\nWichtigste Wörter im ersten Dokument (nach TF-IDF):")
print(df_tfidf.sort_values(by="tfidf", ascending=False).head(5))


# ==========================================
# ANSATZ 2: Word Embeddings (Semantisch / Word2Vec)
# ==========================================
print("\n--- Start Word2Vec Training (Gensim) ---")

# Word2Vec braucht eine Liste von Listen (Tokenisiert)
# Da 'clean_text' ein String ist, splitten wir ihn wieder an den Leerzeichen
tokenized_data = [text.split() for text in df['clean_text']]

# Modell trainieren
# vector_size=100: Jedes Wort wird durch einen Vektor mit 100 Dimensionen dargestellt
# window=5: Kontext ist 5 Wörter links und rechts
# min_count=2: Ignoriere Wörter, die nur 1x vorkommen
w2v_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=2, workers=4)

print("Modell trainiert.")

# TEST: Semantik prüfen
# Wir suchen Wörter, die mathematisch ähnlich zu typischen Begriffen sind
test_words = ['credit', 'money', 'loan', 'service', 'account']

print("\nSemantische Ähnlichkeiten (Beispiele):")
for word in test_words:
    if word in w2v_model.wv:
        similar = w2v_model.wv.most_similar(word, topn=3)
        print(f"Ähnlich zu '{word}': {similar}")
    else:
        print(f"Wort '{word}' nicht im Vokabular (zu selten oder rausgefiltert).")

# Optional: Speichern der Modelle für Schritt 3 (Themenextraktion)
# Wir brauchen später vor allem die TF-IDF Matrix und den Vectorizer für NMF
import pickle
with open("data/tfidf_data.pkl", "wb") as f:
    pickle.dump((tfidf_matrix, tfidf_vectorizer), f)

print("\nFertig. Daten für Phase 3 gespeichert.")