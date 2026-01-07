import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# NLTK-Ressourcen herunterladen (nur beim ersten Mal notwendig)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(filepath, num_rows=None):
    """
    Lädt die Daten. Optional limitiert auf num_rows für schnelleres Testen.
    """
    print(f"Lade Daten aus {filepath}...")
    # 'Consumer Complaint' ist oft der Spaltenname für den Text in diesem Datensatz
    # Wir filtern Zeilen ohne Text direkt raus (dropna)
    df = pd.read_csv(filepath, nrows=num_rows)
    
    # Prüfen, wie die Text-Spalte heißt (oft 'Consumer complaint narrative')
    # Passe 'text_column' an deinen Datensatz an!
    text_column = 'Consumer complaint narrative' 
    
    df = df.dropna(subset=[text_column]).copy()
    print(f"{len(df)} Einträge mit Text geladen.")
    return df, text_column

def clean_text(text):
    """
    Führt die Vorverarbeitung gemäß Konzept durch:
    1. Noise Removal (Sonderzeichen/URLs)
    2. Lowercasing
    3. Tokenisierung
    4. Stopword Removal
    5. Lemmatisierung
    """
    # 1. Noise Removal & 2. Lowercasing
    text = text.lower() # Text in Kleinbuchstaben umwandeln
    text = re.sub(r'http\S+', '', text) # URLs entfernen (vorden Sonderzeichen, um Fehler zu vermeiden) 
    text = re.sub(r'x{2,}', '', text)
    text = re.sub(r'[^a-z\s]', '', text) # Alles außer Kleinbuchstaben und Leerzeichen entfernen, nur Wörter behalten

    # 3. Tokenisierung
    tokens = word_tokenize(text)
    
    # 4. Stopwords laden
    stop_words = set(stopwords.words('english'))
    
    # 5. Lemmatisierung initiieren
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2: # Kurze Schnipsel entfernen
            lemma = lemmatizer.lemmatize(token)
            clean_tokens.append(lemma)
            
    return " ".join(clean_tokens)

# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    # Pfad anpassen!
    FILE_PATH = 'data/complaints.csv' 
    
    # Schritt 1: Laden (zum Testen nur 1000 Zeilen)
    df, text_col = load_data(FILE_PATH, num_rows=5000)
    
    # Schritt 2: Vorverarbeitung anwenden
    print("Starte Vorverarbeitung (das kann kurz dauern)...")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    # Ergebnis prüfen
    print("\nBeispiel Vorher vs. Nachher:")
    print("-" * 50)
    print(f"Original: {df[text_col].iloc[0][:100]}...")
    print(f"Clean:    {df['clean_text'].iloc[0][:100]}...")
    print("-" * 50)
    
    # Optional: Zwischenspeichern als Pickle (schneller als CSV für den nächsten Schritt)
    df.to_pickle("data/corpus_cleaned.pkl")
    print("Daten vorverarbeitet und gespeichert.")
