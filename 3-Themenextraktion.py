import pandas as pd
import pickle
from sklearn.decomposition import NMF
import gensim
from gensim import corpora

# Konfiguration
INPUT_FILE = "data/corpus_cleaned.pkl"
TFIDF_FILE = "data/tfidf_data.pkl"
NUM_TOPICS = 5  # Wie viele Themen wollen wir finden?

# 1. Daten laden
print("Lade Daten...")
df = pd.read_pickle(INPUT_FILE)
# Sicherstellen, dass keine leeren Texte dabei sind
df = df[df['clean_text'].str.len() > 2].reset_index(drop=True)

with open(TFIDF_FILE, "rb") as f:
    tfidf_matrix, tfidf_vectorizer = pickle.load(f)

print(f"Analysebasis: {len(df)} Dokumente.")

# ==========================================
# ANSATZ A: NMF (Non-negative Matrix Factorization)
# Bibliothek: scikit-learn
# ==========================================
print("\n--- Start NMF (Scikit-Learn) ---")

# Modell trainieren
nmf_model = NMF(n_components=NUM_TOPICS, random_state=42, max_iter=500)
nmf_model.fit(tfidf_matrix)

# Hilfsfunktion zur Anzeige der Themen
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        # Die Top-Wörter für dieses Thema holen
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics[f"Thema {topic_idx+1}"] = ", ".join(top_features)
        print(f"Thema {topic_idx+1}: {', '.join(top_features)}")
    return topics

print("NMF Themen:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(nmf_model, tfidf_feature_names, 5)


# ==========================================
# ANSATZ B: LDA (Latent Dirichlet Allocation)
# Bibliothek: Gensim (wie im Konzept versprochen)
# ==========================================
print("\n--- Start LDA (Gensim) ---")

# Gensim benötigt tokenisierte Listen, keine Strings
tokenized_text = [text.split() for text in df['clean_text']]

# 1. Wörterbuch erstellen (Mapping Wort -> ID)
dictionary = corpora.Dictionary(tokenized_text)

# Optional: Extrem seltene und extrem häufige Wörter filtern
dictionary.filter_extremes(no_below=2, no_above=0.9)

# 2. Korpus erstellen (Bag of Words Format)
corpus = [dictionary.doc2bow(text) for text in tokenized_text]

# 3. Modell trainieren
lda_model = gensim.models.LdaMulticore(corpus=corpus, 
                                       id2word=dictionary, 
                                       num_topics=NUM_TOPICS, 
                                       passes=10, 
                                       workers=2,
                                       random_state=42)

# Themen anzeigen
print("LDA Themen:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Thema {idx+1}: {topic}")

# ==========================================
# INTERPRETATION (Automatisch zuweisen)
# ==========================================
print("\n--- Beispiel-Zuweisung ---")
# Wir nehmen das erste Dokument und schauen, wozu es gehört (laut NMF)
doc_index = 0
if len(df) > 0:
    doc_vector = tfidf_matrix[doc_index]
    topic_scores = nmf_model.transform(doc_vector)
    best_topic = topic_scores.argmax()
    
    print(f"Dokument: {df['clean_text'].iloc[doc_index][:80]}...")
    print(f"-> Wurde von NMF dem Thema {best_topic+1} zugeordnet.")