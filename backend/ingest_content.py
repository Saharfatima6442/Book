import os
import asyncio
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
import requests
from main import COLLECTION_NAME, qdrant_client, cohere_client
import sys
sys.path.append("../../AI-Book")

def extract_text_from_md_file(file_path):
    """Extract text content from a markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Convert markdown to HTML, then extract text
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()

    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # If we're not at the end and the chunk ends mid-sentence, try to end at a sentence boundary
        if end < len(text):
            # Look for a sentence boundary near the end of the chunk
            for i in range(min(len(chunk), 50), 0, -1):
                if chunk[i] in '.!?':
                    chunk = chunk[:i+1]
                    end = start + i + 1
                    break

        chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)

        # Make sure we don't get stuck in a loop
        if start == end:
            break

    return chunks

def get_all_md_files(directory):
    """Recursively get all markdown files in a directory"""
    md_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md') or file.endswith('.mdx'):
                md_files.append(os.path.join(root, file))
    return md_files

def ingest_book_content():
    """Ingest all book content into the vector database"""
    print("Starting book content ingestion...")

    # Get all markdown files from the docs directory
    docs_path = "../AI-Book/docs"  # Relative to backend directory
    md_files = get_all_md_files(docs_path)

    print(f"Found {len(md_files)} markdown files to process")

    documents_to_ingest = []

    for file_path in md_files:
        print(f"Processing {file_path}...")

        try:
            # Extract text content from the file
            content = extract_text_from_md_file(file_path)

            # Get the relative path as source
            source = os.path.relpath(file_path, "../AI-Book")

            # Extract title from filename or first line
            title = os.path.basename(file_path).replace('.md', '').replace('.mdx', '').replace('-', ' ').title()

            # Extract chapter from directory structure
            chapter = os.path.dirname(source).split(os.sep)[-1] if os.sep in source else "Introduction"

            # Chunk the content
            chunks = chunk_text(content)

            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    document = {
                        "content": chunk,
                        "metadata": {
                            "source": source,
                            "title": title,
                            "chapter": chapter,
                            "chunk_index": i
                        }
                    }
                    documents_to_ingest.append(document)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print(f"Prepared {len(documents_to_ingest)} document chunks for ingestion")

    # Batch process documents to avoid memory issues
    batch_size = 10
    for i in range(0, len(documents_to_ingest), batch_size):
        batch = documents_to_ingest[i:i+batch_size]
        print(f"Ingesting batch {i//batch_size + 1}/{(len(documents_to_ingest)-1)//batch_size + 1}...")

        # Prepare documents for the API
        from pydantic import BaseModel
        from typing import List, Dict

        class Document(BaseModel):
            content: str
            metadata: Dict

        api_docs = [Document(content=doc["content"], metadata=doc["metadata"]) for doc in batch]

        # Call the ingest endpoint (you'll need to run the FastAPI app first)
        import requests

        response = requests.post(
            "http://localhost:8000/ingest",
            json={"documents": [doc.dict() for doc in api_docs]}
        )

        if response.status_code != 200:
            print(f"Error ingesting batch: {response.text}")
        else:
            print(f"Successfully ingested batch {i//batch_size + 1}")

    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_book_content()