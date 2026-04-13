import requests
import logging as log

class OllamaEmbeddings:
    def __init__(self, model="mxbai-embed-large", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        log.info(f"CLASS:OllamaEmbeddings initialized with model: {model}")

    def execute(self, text):
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            log.error(f"Error in OllamaEmbeddings.execute: {e}")
            print(f"Error: {e}")
            return None

    def get_embedding_dim(self):
        try:
            test_embedding = self.execute("test")
            if test_embedding:
                return len(test_embedding)
        except Exception as e:
            log.error(f"Error getting embedding dimension: {e}")
        return None


class OllamaChat:
    def __init__(self, model="llama3", base_url="http://localhost:11434", temperature=0.0):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.messages = []
        log.info(f"CLASS:OllamaChat initialized with model: {model}")

    def add_message(self, role, content):
        json_message = {
            "role": role,
            "content": content
        }
        self.messages.append(json_message)

    def execute(self):
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": self.temperature
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            log.error(f"Error in OllamaChat.execute: {e}")
            print(f"Error: {e}")
            return None

    def execute_stream(self):
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": self.temperature,
                    "stream": True
                },
                stream=True
            )
            response.raise_for_status()
            full_text = ""
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        chunk = data["message"]["content"]
                        print(chunk, end="", flush=True)
                        full_text += chunk
            self.add_message("assistant", full_text)
            return full_text
        except Exception as e:
            log.error(f"Error in OllamaChat.execute_stream: {e}")
            print(f"Error: {e}")
            return None