# Local helper modules for datamyn
# These provide alternative implementations for local deployment

from .ollama_helper import OllamaEmbeddings, OllamaChat
from .chroma_helper import ChromaDB


class LocalRAG:
    def __init__(
        self,
        embedding_model="mxbai-embed-large",
        chat_model="llama3",
        chroma_persist_dir="./chroma_db",
        db_connection=None
    ):
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.embedding_helper = OllamaEmbeddings(model=embedding_model)
        self.chat_helper = OllamaChat(model=chat_model)
        self.chroma = ChromaDB(persist_directory=chroma_persist_dir)
        self.postgres = None

    def query(self, user_query, namespace, top_k=5, system_prompt=None):
        embed = self.embedding_helper.execute(user_query)
        if not embed:
            return None, []

        results = self.chroma.query(namespace, [embed], n_results=top_k)
        if not results or not results.get('documents'):
            return None, []

        context = []
        source_info = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results.get('metadatas', [{}])[0][i] if results.get('metadatas') else {}
            source = metadata.get('source', 'unknown')
            chunk_num = metadata.get('chunk_number', i + 1)
            score = results.get('distances', [[0.0]])[0][i] if results.get('distances') else 0.0

            context.append(f"Source: {source}, Chunk: {chunk_num}, Content: {doc}")
            source_info.append({
                'source': source,
                'chunk_number': chunk_num,
                'score': score,
                'content': doc
            })

        prompt = f"Context:\n" + "\n\n---\n\n".join(context) + f"\n\nQuestion: {user_query}\nAnswer:"
        
        if system_prompt:
            self.chat_helper.add_message("system", system_prompt)
        self.chat_helper.add_message("user", prompt)
        
        response = self.chat_helper.execute()
        return response, source_info


def create_prompt_local(query, context_list):
    prompt_start = "Answer based on the context.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    return prompt_start + "\n\n---\n\n".join(context_list) + prompt_end


def create_system_prompt_local(role="helpful assistant"):
    return f"You are a {role}. If you don't know something, say so."