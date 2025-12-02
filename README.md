# ragPDF - PDF Document QnA Bot

This is a **Retrieval-Augmented Generation (RAG)** app that lets you chat with your PDF documents without sending any data to the cloud.

I built this because I wanted to see if I could get a RAG system running entirely on my laptop (an RTX 3050). It uses **Llama 3.2 (3B)** for the model and a custom **Hybrid Search** pipeline to make sure it actually finds the right answers in your files.

## Features

  * **Private:** Everything runs locally using [Ollama](https://ollama.com). You can disconnect your WiFi and it still works.
  * **Better Search:** I didn't just use basic vector search. I implemented a **Hybrid Search** (FAISS + BM25). This means it finds concepts and exact matches much better than standard bots.
  * **Can run on mid-end laptops:** It’s tuned to run on consumer hardware (tested on an RTX 3050 with 4GB VRAM). It uses token streaming so you don't stare at a blank screen while it thinks.
  * **Citations:** It tells you exactly which page of the PDF the answer came from, so you know it's not hallucinating.
  * **Dockerized:** Can work on any machine with the right configuration of the docker run command

## Stack

  * **Model:** Llama 3.2 3B (via Ollama)
  * **Embeddings:** Nomic Embed Text
  * **Backend:** LangChain
  * **Frontend:** Streamlit
  * **Vector DB:** FAISS

-----

## How to Run It

### 1\. Prerequisites (Ollama)

You need Ollama installed to run the LLM.

1.  Download it from [ollama.com](https://ollama.com).
2.  Open your terminal and pull the models needed:
    ```bash
    ollama pull llama3.2:3b

    ollama pull nomic-embed-text
    ```

### 2\. Running the App (Python)

Clone the repo and install the requirements:

```bash
git clone https://github.com/yourusername/local-rag-chatbot.git
cd ragPDF

# Create a virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
```

Now running the app:

```bash
streamlit run app.py
```

It should open a browser window automatically, if it doesn't, go to 'http://localhost:8501' to use the app

-----

## Running with Docker

If you don't want to mess with Python environments, I made a Docker image.

**Note for Linux users:** I use the `--network="host"` flag so the container can talk to the Ollama service running on your actual computer.

1.  **Build it:**

    ```bash
    docker build -t local-rag .
    ```

2.  **Run it:**

    ```bash
    docker run --network="host" local-rag
    ```

-----

## Workflow

1.  **Ingest:** The app reads your PDF and breaks it down into chunks (1000 characters).
2.  **Hybrid Indexing:** It saves these chunks in two places:
      * **FAISS:** For vector math (understanding *meaning*).
      * **BM25:** For keyword matching (finding *exact words*).
3.  **Ensemble Retrieval:** When you ask a question, it searches both indexes and combines the results (50/50 weight) to find the best context.
4.  **Generation:** It sends that context + your chat history to Llama 3.2 to write the final answer.
