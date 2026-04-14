# DATAMYN - Query Interface

DATAMYN is a query interface that retrieves relevant chunks using vector similarity search and generates responses using a chat model. It supports both local (Ollama + ChromaDB + PostgreSQL) and cloud (OpenAI + Pinecone + MongoDB) providers.

Key Features
- Vector Search: Retrieves relevant text chunks based on semantic similarity
- Chat Integration: Uses Ollama (local) or OpenAI (cloud) for response generation
- Configurable Prompts: Customizable system and user prompts
- Interactive UI: Optional Gradio interface for interactive querying

## Git Repositories
- https://github.com/open-qe-automation/texten.git
- https://github.com/open-qe-automation/webtexten.git
- https://github.com/open-qe-automation/chunken.git
- https://github.com/open-qe-automation/datamyn.git

## Related Packages
- https://github.com/open-qe-automation/package.utils.git
- https://github.com/open-qe-automation/package.data.loaders.git
- https://github.com/open-qe-automation/package.helpers.git

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Prerequisites

- Python 3.12 or later
- pip
- For local stack: Ollama, PostgreSQL, ChromaDB
- For cloud stack: OpenAI API key, Pinecone, MongoDB

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/open-qe-automation/datamyn.git
    cd datamyn
    ```

2. **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    For local development, use dev-requirements.txt:
    ```bash
    pip install -r dev-requirements.txt
    ```

## Usage

**Command-line query:**
```bash
python command.py "your query here"
```

**Interactive UI:**
```bash
python app.py
```

### Local Stack Setup

1. **Start PostgreSQL:**
   ```bash
   # Using the provided script
   ./start_db.sh
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   # Pull the embedding model
   ollama pull nomic-embed-text
   # Pull the chat model
   ollama pull llama3.1
   ```

### Configuration

The configuration is managed through a `config.json` file (not .env):

```json
{
    "embedding_provider": "ollama",
    "embedding_model": "nomic-embed-text",
    "embedding_base_url": "http://localhost:11434",
    "chat_provider": "ollama",
    "chat_model": "llama3.1",
    "chat_temperature": 0.0,
    "namespace": "banking",
    "top_k": 5,
    "vector_db_provider": "chroma",
    "chroma_persist_directory": "../share/chroma_db",
    "metadata_db_provider": "postgresql",
    "postgresql_connection_string": "postgresql://postgres:ragpassword@localhost:5432/rag",
    "database": "rag-system"
}
```

#### Provider Options

**Embedding Provider:**
- `ollama` (local): Uses Ollama with nomic-embed-text model
- `openai` (cloud): Uses OpenAI with text-embedding-3-small model

**Chat Provider:**
- `ollama` (local): Uses Ollama with llama3.1 model
- `openai` (cloud): Uses OpenAI with gpt-4o-mini model

**Vector DB Provider:**
- `chroma` (local): Persists to local directory
- `pinecone` (cloud): Requires PINECONE_API_KEY env var

**Metadata DB Provider:**
- `postgresql` (local): Uses PostgreSQL connection string
- `mongodb` (cloud): Requires MONGO env var