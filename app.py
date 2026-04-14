# datamyn - DATAMYN Query Interface
import os
import json
import gradio as gr

# Import abstraction managers from msuliot.helpers
from msuliot.embedding_manager import create_embedding_manager
from msuliot.vector_db_manager import create_vector_db_manager
from msuliot.metadata_db_manager import create_metadata_db_manager

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize managers
embedding_manager = create_embedding_manager(config)
vector_db_manager = create_vector_db_manager(config)
metadata_db_manager = create_metadata_db_manager(config)


def create_system_prompt():
    return "You are honest, professional and positive. If there's something you don't know then just say so."


def create_prompt(query, context_list):
    prompt_start = "Answer based on the context.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    return prompt_start + "\n\n---\n\n".join(context_list) + prompt_end


# main function
def main(namespace, system_prompt, query):
    print("Start: Main function")
    
    # Get embedding for user query
    embed_result = embedding_manager.execute(query)
    if not embed_result:
        return "Error: Could not generate embedding", "[]"
    
    # Extract embedding vector
    if hasattr(embed_result, 'data'):
        embedding = embed_result.data[0].embedding
    else:
        embedding = embed_result.get('embedding', [])
    
    # Query vector DB
    top_k = config.get('top_k', 5)
    # Use namespace from Gradio dropdown (line 48 removed - was overwriting with config value)
    vector_result = vector_db_manager.query(
        query_embedding=embedding,
        top_k=top_k,
        namespace=namespace
    )
    
    if not vector_result or not vector_result.get('documents'):
        return "No matching documents found", "[]"
    
    context = []
    source_info = []
    
    # Process matches
    documents = vector_result.get('documents', [[]])[0]
    metadatas = vector_result.get('metadatas', [[]])[0]
    distances = vector_result.get('distances', [[]])[0]
    
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        source_file = metadata.get('source', 'unknown')
        chunk_number = metadata.get('chunk_number', i + 1)
        score = distances[i] if i < len(distances) else 0.0
        
        # Get full text from metadata DB
        chunk_id = metadata.get('chunk_id', f'chunk_{i}')
        table_name = config.get('database', 'rag-system')
        
        chunk_data = metadata_db_manager.get_chunk_by_id(table_name, chunk_id)
        if chunk_data:
            text = chunk_data.get('text', doc)
        else:
            text = doc
        
        context.append(f"SourceFile: {source_file}, Chunk: {chunk_number}, Content: {text}")
        source_info.append({
            'source': source_file,
            'chunk_number': chunk_number,
            'score': float(score),
            'content': text[:200] + '...' if len(text) > 200 else text
        })
    
    # Generate prompt
    prompt = create_prompt(query, context)
    
    # Get LLM response
    from msuliot.ollama_helper import OllamaChat
    
    chat_model = config.get('chat_model', 'llama3')
    chat_temp = config.get('chat_temperature', 0.0)
    chat_base_url = config.get('embedding_base_url', 'http://localhost:11434')
    
    oaic = OllamaChat(model=chat_model, base_url=chat_base_url, temperature=chat_temp)
    oaic.add_message("system", system_prompt)
    oaic.add_message("user", prompt)
    response = oaic.execute()
    
    if not response:
        response = "Error: Could not generate response"
    
    return response, json.dumps(source_info, indent=2)


# Gradio UI
gr.close_all()

# Get available namespaces from ChromaDB (dynamic - not hardcoded)
try:
    collections = vector_db_manager.client.list_collections()
    dropdown_namespaces = [c.name for c in collections]
    if not dropdown_namespaces:
        dropdown_namespaces = [config.get('namespace', 'banking')]
except Exception as e:
    print(f"Could not list collections: {e}")
    dropdown_namespaces = [config.get('namespace', 'banking')]

with gr.Blocks() as demo:
    gr.Markdown("# DATAMYN - RAG Query Interface")
    gr.Markdown("Query your documents using local LLM (Ollama), ChromaDB, and PostgreSQL")
    
    with gr.Row():
        with gr.Column():
            namespace = gr.Dropdown(
                label="Choose a namespace",
                choices=dropdown_namespaces,
                value=dropdown_namespaces[0] if dropdown_namespaces else "banking"
            )
            system_prompt = gr.Textbox(
                label="System Prompt",
                lines=3,
                value="I'm an AI assistant. If I don't know something, I'll be upfront about it."
            )
            user_prompt = gr.Textbox(
                label="Ask a question about your documents",
                lines=5,
                placeholder="What would you like to know?"
            )
            submit_btn = gr.Button("Submit", variant="primary")

        response = gr.Markdown(label="Response")
        source = gr.components.JSON(label="Source Documents")

    submit_btn.click(
        fn=main,
        inputs=[namespace, system_prompt, user_prompt],
        outputs=[response, source]
    )

demo.launch(server_name="localhost", server_port=8765)