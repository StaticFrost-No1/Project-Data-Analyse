import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import os
import sys
import gdown

# ==========================================
# KONFIGURATION
# ==========================================
DATA_DIR = 'data'
CSV_FILENAME = 'complaints.csv'
INPUT_FILE = os.path.join(DATA_DIR, CSV_FILENAME)
OUTPUT_FILE = os.path.join(DATA_DIR, 'corpus_cleaned.pkl')
TEXT_COLUMN = 'Consumer complaint narrative' 

# Link zur Kopie der Datensammlung
DATABASE_URL = "https://drive.google.com/file/d/1rHbDFfR2FeU1P02_PbQ1jvpGzdG68IEH/view?usp=drive_link"

# ==========================================
# FUNKTIONEN
# ==========================================
def setup_nltk():
    """
    Lädt NLTK-Ressourcen herunter (silent mode).
    """
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Lade NLTK-Ressourcen herunter...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

setup_nltk() # führt das Setup direkt aus, damit die Variablen darunter sicher befüllt werden können.
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def load_database():
    """
    Lädt die Datenbank herunter.
    Zeigt einen Fehler, wenn dies fehlschlägt.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"Datei '{CSV_FILENAME}' nicht gefunden.")
        print(f"Starte Download von Google Drive...")
        try:
            # gdown löst Probleme beim herunterladen größerer Dateien von Google Drive 
            gdown.download(DATABASE_URL, INPUT_FILE, quiet=False, fuzzy=True)
        except Exception as e:
            print(f"Download fehlgeschlagen: {e}")
            return
    else:
        print(f"Datei '{CSV_FILENAME}' ist vorhanden, weiter gehts.")

def clean_text(text):
    """
    Preprocessing Pipeline: 
    Lowercasing -> Noise Removal -> Tokenization -> Stopword Removal & Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercasing
    text = text.lower()

    # 2. Noise Removal (An diesen Datensatz angepasst)
    text = re.sub(r'http\S+', '', text) # Entfernt URLs
    text = re.sub(r'x{2,}', '', text) # Entfernt "xxxx", die im Datensatz zur Anonymisierung genutzt werden
    text = re.sub(r'[^a-z\s]', '', text) # Entfernt alles außer Buchstaben

    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal & Lemmatization
    clean_tokens = []
    for token in tokens:
        # Nur Wörter länger als 2 Zeichen behalten ("is", "to")
        if token not in STOP_WORDS and len(token) > 2:
            lemma = LEMMATIZER.lemmatize(token)
            clean_tokens.append(lemma)
            
    return " ".join(clean_tokens)

# ==========================================
# MAIN
# ==========================================
def main():
    print("==========================================")
    print("   PHASE 1: Daten-Download & Preprocessing")
    print("==========================================")

    # 1. Ordner erstellen
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 2. Downloads
    load_database()

    # 3. Daten laden
    print(f"\n1. Lade Daten aus '{INPUT_FILE}'...")
    if not os.path.exists(INPUT_FILE): # Wenn der Download schiefging, bricht das Programm ab
        print("FEHLER: Datei fehlt. Bitte Download prüfen.")
        return
    try:
        df = pd.read_csv(INPUT_FILE, usecols=[TEXT_COLUMN])
    except Exception as e:
        print(f"Konnte CSV nicht lesen: {e}")
        return

    # 4. Cleaning
    print("2. Starte Textbereinigung.")
    print("- Das kann einige Minuten dauern...")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str) # Sicherstellen, dass alles String ist
    df['clean_text'] = df[TEXT_COLUMN].apply(clean_text) # Apply
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True) # Leere Zeilen rauswerfen
    
    # 5. Speichern
    print(f"\n3. Speichere verarbeitete Daten...")
    df.to_pickle(OUTPUT_FILE)
    print(f"Fertig! Gespeichert in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()