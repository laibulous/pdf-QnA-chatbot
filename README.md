# AI-Powered PDF Question Answering Chatbot

A Retrieval-Augmented Generation (RAG) pipeline that allows users to chat with their PDF documents. This project uses a hybrid architecture, combining **local embeddings** for privacy and speed with **Google's Gemini Pro** for high-quality natural language reasoning.

## Features
- **RAG Architecture:** accurately retrieves context from large documents to prevent LLM hallucinations.
- **Hybrid Search:** Uses `HuggingFace` models locally for zero-cost embedding generation.
- **Vector Database:** Implements **ChromaDB** for efficient similarity search and retrieval.
- **State-of-the-Art LLM:** Integrates **Google Gemini Flash Latest** for fast and accurate answer generation.

## Tech Stack
- **Language:** Python 3.11
- **Orchestration:** LangChain
- **LLM:** Google Gemini API (`gemini-flash-latest`)
- **Vector Store:** ChromaDB
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers)

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/laibulous/pdf-QnA-chatbot.git](https://github.com/laibulous/pdf-QnA-chatbot.git)
   cd pdf-QnA-chatbot

```

2. **Create a Virtual Environment**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```


4. **Get a Google API Key**
* Visit [Google AI Studio](https://aistudio.google.com/) to get your free API key.



## Usage

1. Place your target PDF file in the root directory and rename it to `your_document.pdf` (or update the filename in `main.py`).
2. Run the application:
```bash
python main.py

```


3. Enter your API key when prompted (input is hidden for security).
4. The system will index the document and generate an answer based on your query.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

```
