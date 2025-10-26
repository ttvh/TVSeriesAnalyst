# AI-Powered TV Series Analysis System (Naruto)

This project is a comprehensive, multi-faceted AI/NLP system designed to analyze the entire "Naruto" TV series. It extracts, processes, and analyzes data from subtitles, character profiles, and external wikis to build a suite of powerful analytical tools.

The project combines several advanced NLP pipelines into a single, interactive **Gradio** web application, allowing users to explore:
1.  **Zero-Shot Theme Analysis:** Analyze and visualize the prevalence of complex themes (e.g., "friendship," "revenge," "sacrifice") across the entire series.
2.  **Character Network Analysis:** Use Named Entity Recognition (NER) to build and visualize the relationships and interaction networks between characters.
3.  **Jutsu Classifier:** Train a custom `transformers` model to classify any "Jutsu" (special ability) into its proper type (e.g., Ninjutsu, Genjutsu, Taijutsu).
4.  **Character Chatbot:** Implement a fine-tuned, instruction-based language model (LLM) that allows users to "chat" with their favorite characters from the series.

---

## 1. Core Components & Technologies

* **Web Scraping:** Used `Scrapy` to build a spider that crawls wiki pages, extracting structured data on Jutsu descriptions, types, and character information.
* **Data Processing:** `Pandas` and `Numpy` for extensive data wrangling, cleaning (e.t., removing HTML tags), and feature engineering.
* **NLP/ML Pipelines:** `Hugging Face Transformers`, `Spacy`, and `Scikit-learn`.
* **Vector Database:** `ChromaDB` (implied, for semantic search capabilities).
* **Model Training & Fine-Tuning:** `Hugging Face Trainer`, `TRL` (Transformer Reinforcement Learning), `PyTorch`.
* **Web Application:** `Gradio` for building the interactive dashboard.

---

## 2. Project Pipeline Breakdown

This project is composed of four primary AI systems.

### a. Pipeline 1: Zero-Shot Theme Analysis 
This pipeline analyzes the script/subtitles of the entire series to find out which themes are most prominent.

1.  **Model:** Uses a pre-trained **Zero-Shot Classifier** (`facebook/bart-large-mnli`) from Hugging Face.
2.  **Data:** Processes the show's subtitles (`.srt` files) by cleaning and formatting the text.
3.  **Inference:** The model is run in batches over the subtitle data. It classifies text segments against a flexible list of themes (e.g., "betrayal," "loneliness," "teamwork") without needing to be pre-trained on them.
4.  **Visualization:** The results are aggregated using `Pandas` to calculate mean scores for each theme, which are then plotted in a `Gradio` bar chart.

### b. Pipeline 2: Character Network (NER) 
This system identifies which characters interact in the show and builds a relationship graph.

1.  **Model:** Uses a **Named Entity Recognition (NER)** model from `Spacy` (or Hugging Face) to identify character names within the text.
2.  **Entity Extraction:** The pipeline scans sentences to extract all named entities (characters).
3.  **Relationship Building:** It creates co-occurrence pairs (e.g., if "Naruto" and "Sasuke" appear in the same sentence, a relationship is logged).
4.  **Standardization:** The relationship pairs are sorted alphabetically (e.g., "Naruto|Sasuke") to ensure "Naruto-Sasuke" and "Sasuke-Naruto" are counted as the same relationship.
5.  **Visualization:** The aggregated relationships are used to generate a character network graph, showing who interacts with whom most frequently.

### c. Pipeline 3: Custom Jutsu Classifier 
This pipeline involves training a **custom text classification model** to categorize any Jutsu description into its correct type.

1.  **Data:** Uses the data scraped by `Scrapy` (Jutsu name, description, and type).
2.  **Preprocessing:** A `cleaner` function is built to remove HTML tags and artifacts from the scraped text.
3.  **Tokenization:** The text data is tokenized and converted to a numerical format using a Hugging Face `AutoTokenizer`.
4.  **Class Imbalance:** `compute_class_weight` is used to handle the imbalance between Jutsu types (e.g., many Ninjutsu, few Genjutsu).
5.  **Training:** A custom `Trainer` from Hugging Face is used to train a `SequenceClassification` model. Class weights are passed to the custom trainer to ensure the model trains effectively on the imbalanced data.
6.  **Inference:** The final, trained model is saved and used in a function that can take new text and predict its class.

### d. Pipeline 4: Character Chatbot 
This is a fine-tuning pipeline that creates a generative AI chatbot capable of role-playing as a specific character.

1.  **Model:** Uses a state-of-the-art, instruction-tuned model (like Llama, Mistral, or Gemma) via the `transformers` library.
2.  **Data:** A custom dataset is built by formatting character-specific lines from the subtitles. This involves creating "system prompts" and "messages" (e.g., `system_prompt: "You are Naruto Uzumaki..."`, `message: "Believe it!"`).
3.  **Fine-Tuning:** The `TRL` (Transformer Reinforcement Learning) library and a `SFTTrainer` (Supervised Fine-Tuning Trainer) are used to fine-tune the pre-trained model on this new, character-specific dataset.
4.  **Quantization:** Configuration is set up to load the model with quantization (e.g., `BitsAndBytesConfig`) to reduce memory usage.
5.  **Inference Pipeline:** A `text-generation` pipeline is created, which takes a user's message and the character's system prompt to generate a in-character response.

---

## 4. Final Application

All four pipelines are integrated into a multi-tabbed `Gradio` dashboard (`dashboard.launch()`), providing a central hub for all analysis tools. The project also covers deployment best practices, such as using `.env` files for API keys (`HuggingFace_Token`) and sharing the app on Google Colab or Hugging Face Spaces.
