# Consumer Complaints Analysis

Dieses Repository enthält die Codebasis für das Portfolio im Bereich Data Science / Data Analysis. 
Ziel des Projekts ist der Vergleich verschiedener NLP-Verfahren zur Vektorisierung und Themenmodellierung sowie die Entwicklung einer optimierten Pipeline zur Verarbeitung unstrukturierter Textdaten aus dem "Consumer Complaints" Datensatz.
Dabei sollen automatisch die Hauptbeschwerdegründe der enthaltenen Kundenbeschwerden identifiziert werden.

## Projektübersicht

Das Projekt vergleicht verschiedene Vektorisierungs- und Modellierungsansätze, um eine robuste Pipeline für Kurztexte zu entwickeln:

* **Datenbasis:** Consumer Complaints Dataset (Finanzbeschwerden)
* **Vektorisierung:** Vergleich von **TF-IDF** (statistisch) und **Word2Vec** (semantisch)
* **Topic Modeling:** Vergleich von **Latent Dirichlet Allocation (LDA)** und **Non-negative Matrix Factorization (NMF)**

## Ergebnisse

Für die Reproduzierbarkeit wurde ein Seed eingefügt, um sicherzustellen, dass bei der Anwendung von Grid-Search immer die gleichen Daten geladen werden. 
Die Analyse zeigte signifikante Unterschiede zwischen den Verfahren:

1.  **Identifizierte Pipeline (TF-IDF & NMF):** Die Kombination aus **TF-IDF** und **NMF** erwies sich als einziger robuster Ansatz. Während Word2Vec keine kohärenten Cluster für die Themenmodellierung lieferte und LDA zu thematischem Rauschen neigte, erzeugte NMF die mit Abstand trennschärfsten Ergebnisse.2.  **Optimale Themenzahl:** Durch Kohärenz-Analysen auf Teildatensätzen (20-80%) wurde **$K=5$** als optimales Cluster-Setup ermittelt.
3.  **Identifizierte Themen:**
    * Inkasso & Schulden (*Debt Collection*)
    * Credit Reporting (*Fehlerhafte Einträge*)
    * Kredite & Hypotheken (*Loans/Mortgages*)
    * Identitätsdiebstahl (*Theft/Fraud*)
    * Rechtliche Beschwerden (*Legal/Regulatory*)

## 🛠 Installation

Dieses Projekt wurde mit **Python 3.11.9** entwickelt. 
Um Kompatibilitätsprobleme zu vermeiden, wird die Verwendung von Linux in Kobination mit `pyenv` zur Verwaltung der Python-Version empfohlen.

1. **Virtuelle Umgebung installieren (empfohlen)**

Falls `pyenv` noch nicht installiert ist, folge bitte dieser Anleitung:

<details>
<summary><strong>Linux</strong></summary>
```bash
curl https://pyenv.run | bash
```
Folge den Bildschirmanweisungen, um pyenv zur Shell hinzuzufügen.
</details>

2. **Repository klonen:**

    ```bash
    git clone https://github.com/StaticFrost-No1/Project-Data-Analysis.git
    cd Project-Data-Analysis
    ```

3. **Python 3.11.9 installieren**

    ```bash
    pyenv install 3.11.9
    pyenv local 3.11.9
    ```

4. **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**

    ```bash
    pyenv exec python -m venv .venv
    source .venv/bin/activate
    ```

5. **Abhängigkeiten installieren:**

    ```bash
    pip install -r requirements.txt
    ```

## Nutzung 

Der Code wurde für die kontrollierte Verwendung in vier Abschnitte unterteilt.
Die Pipeline sollte von Schritt 1-4 nacheinander ausgeführt werden. Ergebnisse können direkt verglichen werden
Phase 3 und 4 sind sehr rechenintensiv und enthalten Parameter, die eine manuelle Anpassung der Balance zwischen Rechenzeit und Präzision 
an die eigenen Bedürfnisse ermöglichen.

### Pipeline

Die Skripte sollten idealerweise in folgender Reihenfolge ausgeführt werden:

1.  **Preprocessing:** Bereinigung der Rohdaten.
    ```bash
    python 1_preprocessing.py
    ```
2.  **Vektorisierung:** Erstellung der TF-IDF und Word2Vec Modelle.
    ```bash
    python 2_vectorization.py
    ```
3.  **Kohärenz-Berechnung:** Suche nach dem optimalen $K$.
    ```bash
    python 3_optimization_coherence.py
    ```
4.  **Themen-Modellierung:** Generierung der Themen mit NMF.
    ```bash
    python 4_final_topic_modeling.py
    ```

### Parameter

die eine manuelle anpassung der Größe der verarbeiteten Teildatensätze, verwendeten CPU-Kerne, 
und Anzahl der Durchläufe ermöglichen, 

- **SAMPLE_FRAC:** Sampling-Rate passt die Größe des verarbeiteten Teildatensatzes für an (0.60 = 60% der Daten)
    - Ein höherer Wert steigert zwar die Präzision, erhöht aber Rechenzeit und Speicherbedarf signifikant
    - Stürzt das Programm ab, sollte dieser Wert nach unten angepasst werden.
    - Ein Wert von `0.60` hat sich als idealer Kompromiss zwischen Präzision und Ressourcenverbrauch erwiesen
- **NUM_TOPICS:** Empfohlene Themenzahl K (nur bei der Themen-Modellierung)
- **WORKERS:** Zahl der verwendeten CPU-Kerne (mehr Kerne benötigen mehr RAM)
- **PASSES:** Anzahl der kompletten Durchläufe (empfohlen: 10)

## Lizenz

Dieses Projekt ist unter der [MIT License](LICENSE) lizenziert.