import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_system_prompt():
    return "You are honest, professional and positive. If you don't know, say so."


def create_prompt(query, res):
    prompt_start = "Answer based on context.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    return prompt_start + "\n\n---\n\n".join(res) + prompt_end


class TestCreatePrompt:
    def test_create_system_prompt(self):
        prompt = create_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Honest" in prompt or "honest" in prompt.lower()

    def test_create_prompt_with_context(self):
        query = "What is this project?"
        context = ["Context 1: The project is a RAG system.", "Context 2: It uses local LLM."]
        prompt = create_prompt(query, context)
        assert query in prompt
        assert "Context:" in prompt

    def test_create_prompt_empty_context(self):
        query = "Test query"
        context = []
        prompt = create_prompt(query, context)
        assert query in prompt
        assert "Question:" in prompt


class TestLocalHelpers:
    def test_ollama_import(self):
        from helpers.ollama_helper import OllamaEmbeddings, OllamaChat
        assert OllamaEmbeddings is not None
        assert OllamaChat is not None

    def test_chroma_import(self):
        from helpers.chroma_helper import ChromaDB
        assert ChromaDB is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])