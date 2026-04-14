# datamyn - DATAMYN Query Interface
import os
import json
import gradio as gr

# Import abstraction managers from msuliot.helpers
from msuliot.embedding_manager import create_embedding_manager
from msuliot.vector_db_manager import create_vector_db_manager
from msuliot.metadata_db_manager import create_metadata_db_manager
from msuliot.chat_manager import create_chat_manager

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize managers
embedding_manager = create_embedding_manager(config)
vector_db_manager = create_vector_db_manager(config)
metadata_db_manager = create_metadata_db_manager(config)
chat_manager = create_chat_manager(config)


def create_system_prompt():
    return "You are honest, professional and positive. If there's something you don't know then just say so."


def create_prompt(query, context_list):
    prompt_start = "Answer based on the context.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    return prompt_start + "\n\n---\n\n".join(context_list) + prompt_end


# main function
def main(namespace, system_prompt, query):
    print(f"Start: Main function")
    print(f"  Raw namespace from Gradio: {repr(namespace)}")
    print(f"  system_prompt: {repr(system_prompt[:50])}")
    print(f"  query: {repr(query[:50])}")
    
    # Get embedding for user query
    embed_result = embedding_manager.execute(query)
    if not embed_result:
        return "Error: Could not generate embedding", "[]"
    
    # Extract embedding vector (embed_result is a dict, not an object)
    if isinstance(embed_result, dict):
        if 'data' in embed_result and embed_result['data']:
            embedding = embed_result['data'][0]['embedding']
        elif 'embedding' in embed_result:
            embedding = embed_result['embedding']
        else:
            embedding = []
    else:
        embedding = []
    
    # Debug: check embedding
    if not embedding or len(embedding) == 0:
        print("ERROR: Empty embedding returned")
        return "Error: Empty embedding returned by embedding model", "[]"
    
    print(f"Embedding generated, length: {len(embedding)}, first 3: {embedding[:3]}")
    
    # Query vector DB
    top_k = config.get('top_k', 5)
    # Use namespace from Gradio dropdown or default to config
    namespace = namespace or config.get('namespace', 'banking')
    print(f"Raw namespace from Gradio: {repr(namespace)}")
    print(f"Using namespace: '{namespace}'")
    
    # Debug: check what collections exist
    try:
        collections = vector_db_manager.client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
    except Exception as e:
        print(f"Error listing collections: {e}")
    
    vector_result = vector_db_manager.query(
        query_embedding=embedding,
        top_k=top_k,
        namespace=namespace
    )
    
    if not vector_result or not vector_result.get('matches'):
        print(f"DEBUG: vector_result = {vector_result}")
        return "No matching documents found", "[]"
    
    context = []
    source_info = []
    
    # Process matches from vector DB result
    matches = vector_result.get('matches', [])
    print(f"DEBUG: Processing {len(matches)} matches")
    
    for i, match in enumerate(matches):
        chunk_id = match.get('id', f'chunk_{i}')
        metadata = match.get('metadata', {})
        source_file = metadata.get('source', 'unknown')
        chunk_number = metadata.get('chunk_number', i + 1)
        score = match.get('score', 0.0)
        
        # Get full text from metadata DB
        table_name = config.get('database', 'rag-system')
        
        print(f"DEBUG: Looking up chunk_id: {chunk_id[:30]}...")
        chunk_data = metadata_db_manager.get_chunk_by_id(table_name, chunk_id)
        
        if chunk_data:
            text = chunk_data.get('text', '')
            print(f"DEBUG: Found text: {text[:50]}...")
        else:
            text = match.get('values', '')  # fallback to vector text
            print(f"DEBUG: No PostgreSQL data, using vector text: {text[:50] if text else 'empty'}...")
        
        context.append(f"SourceFile: {source_file}, Chunk: {chunk_number}, Content: {text[:300]}")
        source_info.append({
            'source': source_file,
            'chunk_number': chunk_number,
            'score': float(score),
            'content': text[:200] + '...' if len(text) > 200 else text
        })
    
    print(f"DEBUG: Built context with {len(context)} entries")
    
    # Generate prompt
    prompt = create_prompt(query, context)
    
    # Get LLM response using chat manager (supports both Ollama and OpenAI)
    chat_manager.add_message("system", system_prompt)
    chat_manager.add_message("user", prompt)
    response = chat_manager.execute()
    
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