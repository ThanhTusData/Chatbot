import json
import argparse
from pathlib import Path
import logging

from retrieval.embed import EmbeddingGenerator
from retrieval.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_kb_data(kb_dir: str):
    kb_path = Path(kb_dir)
    documents = []
    
    for file_path in kb_path.glob("*.jsonl"):
        logger.info(f"Loading {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)
    
    return documents

def build_index(kb_dir: str, output_dir: str, embedding_model: str):
    logger.info("Loading knowledge base documents...")
    documents = load_kb_data(kb_dir)
    logger.info(f"Loaded {len(documents)} documents")
    
    logger.info("Initializing embedding model...")
    embedder = EmbeddingGenerator(embedding_model)
    
    logger.info("Generating embeddings...")
    texts = [doc.get('content', doc.get('question', '')) for doc in documents]
    embeddings = embedder.encode(texts, batch_size=32)
    
    logger.info("Building vector store...")
    vectorstore = VectorStore(embedding_dim=embedder.embedding_dim)
    vectorstore.add_documents(embeddings, documents)
    
    logger.info(f"Saving index to {output_dir}...")
    vectorstore.save(output_dir)
    
    logger.info("Index built successfully!")

def main():
    parser = argparse.ArgumentParser(description='Build knowledge base index')
    parser.add_argument('--kb-dir', type=str, default='data/kb', help='Knowledge base directory')
    parser.add_argument('--output', type=str, default='indexes/kb', help='Output directory')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    
    args = parser.parse_args()
    build_index(args.kb_dir, args.output, args.model)

if __name__ == '__main__':
    main()