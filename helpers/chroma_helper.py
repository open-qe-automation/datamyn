import chromadb
from chromadb.config import Settings
import logging as log

class ChromaDB:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            log.info(f"ChromaDB initialized at: {persist_directory}")
        except Exception as e:
            log.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None

    def get_or_create_collection(self, name):
        if self.client:
            return self.client.get_or_create_collection(name)
        return None

    def add_vectors(self, collection_name, ids, embeddings, metadatas=None, documents=None):
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            log.info(f"Added {len(ids)} vectors to collection: {collection_name}")
            return True
        except Exception as e:
            log.error(f"Error adding vectors: {e}")
            return False

    def query(self, collection_name, query_embeddings, n_results=5, where=None):
        try:
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            log.error(f"Error querying vectors: {e}")
            return None

    def delete_collection(self, name):
        try:
            if self.client:
                self.client.delete_collection(name)
                log.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            log.error(f"Error deleting collection: {e}")
            return False