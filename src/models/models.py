import os
import yaml
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import OpenAI
from langchain_openai import AzureOpenAI
from langchain_ollama import OllamaEmbeddings  
from transformers import AutoTokenizer
from langchain.text_splitter import TokenTextSplitter
from typing import Dict, Any
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path):
        """
        Initialize ModelManager by loading configuration and setting up models.
        :param config_path: Path to the YAML configuration file.
        """
        load_dotenv()
        
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully from %s", config_path)
        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

        self.llm = None
        self.embedding_model = None
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        self.max_tokens = 1000  # Conservative limit
        
        self.text_splitter = TokenTextSplitter(
            chunk_size=1000,  # slightly less than max tokens
            chunk_overlap=100
        )
        
        if self.config['llm_provider'] == 'groq':
            self.load_llm_groq()
            
        if self.config['llm_provider'] == 'ollama':
            self.load_llm_ollama()
            
        if self.config['llm_provider'] == 'openai':
            self.load_llm_openai()
        
        
    # def get_llm_response(self):
    #     if self.config['llm_provider'] == 'groq':
    #         return self.llm_groq.invoke(prompt).content
    #     if self.config['llm_provider'] == 'ollama':
    #         return self.llm_ollama(prompt)
    #     if self.config['llm_provider'] == 'openai':
    #         return self.llm_openai(prompt)

    def load_llm_groq(self):
        """
        Lazily loads the LLM Groq model based on the configuration if it hasn't been loaded yet.
        :return: The loaded LLM Groq model.
        """
        if not self.llm:
            try:
                logger.info("Loading Groq LLM model...")
                self.llm = ChatGroq(
                    model_name=self.config['llm_model'],
                    api_key=os.getenv("GROQ_API_KEY")
                )
                logger.info("Groq LLM model loaded successfully.")
                
            except KeyError as e:
                logger.error("Missing required config key for LLM: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading Groq LLM model: %s", e)
                raise
    
    def load_llm_openai(self):
        """
        Loads the OpenAI model.
        :return: The loaded OpenAI model.
        """
        if not self.llm:
            try:
                logger.info("Loading OpenAI model...")
                # openai.api_key = os.getenv("OPENAI_API_KEY")
                # os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
                # os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
                # os.environ["AZURE_OPENAI_API_KEY"] = "..."
                print("AZURE_OPENAI_DEPLOYMENT:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
                load_dotenv()
                # print("AZURE_OPENAI_DEPLOYMENT:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
                self.llm = AzureOpenAI(
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # The deployment name of the model
                    )
                logger.info("OpenAI model loaded successfully.")
                
            except KeyError as e:
                logger.error("Missing required config key for OpenAI LLM: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading OpenAI LLM model: %s", e)
                raise
            
    def load_llm_ollama(self):
        """
        Loads the Ollama LLM model.
        :return: The loaded Ollama LLM model.
        """
        if not self.llm:
            try:
                logger.info("Loading Ollama LLM model...")
                
                self.llm = ChatOllama(
                                model=self.config['llm_model'],
                                temperature=0,
                                )
                logger.info("Ollama LLM model loaded successfully.")
                          
            except KeyError as e:
                logger.error("Missing required config key for Ollama LLM: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading Ollama LLM model: %s", e)
                raise

    def load_embedding_model(self):
        """
        Lazily loads the Hugging Face embedding model based on the configuration if it hasn't been loaded yet.
        :return: The loaded embedding model.
        """
        if not self.embedding_model:
            try:
                logger.info("Loading embedding model...")
                self.embedding_model = OllamaEmbeddings(model=self.config['embedding_model'])
                logger.info("Embedding model loaded successfully.")
            except KeyError as e:
                logger.error("Missing required config key for embedding model: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading embedding model: %s", e)
                raise
        return self.embedding_model

    def count_tokens_safely(self, text: str) -> int:
        """
        Safely count tokens with fallback options.
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error in primary token counting: {e}")
            try:
                # Fallback to simple estimation
                return len(text.split()) * 1.3
            except:
                return 0

    def process_text_in_chunks(self, text: str) -> Dict[str, Any]:
        """
        Process text in chunks and return token statistics.
        """
        chunks = []
        current_chunk = ""
        current_tokens = 0
        total_tokens = 0
        
        for sentence in text.split('. '):
            sentence_tokens = self.count_tokens_safely(sentence)
            if current_tokens + sentence_tokens > self.max_tokens:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += '. ' + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
            total_tokens += sentence_tokens
            
        if current_chunk:
            chunks.append(current_chunk)
            
        return {
            "chunks": chunks,
            "total_tokens": total_tokens,
            "num_chunks": len(chunks)
        }

    def safe_invoke(self, prompt):
        """
        Safely invoke the LLM by splitting text if needed.
        """
        try:
            # Split the prompt if it's too long
            chunks = self.text_splitter.split_text(prompt)
            
            if len(chunks) == 1:
                # If only one chunk, process normally
                return self.llm.invoke(prompt).content
            
            # If multiple chunks, process each and combine
            responses = []
            for chunk in chunks:
                response = self.llm.invoke(chunk).content
                responses.append(response)
            
            # Combine responses with a final summarization
            combined_response = " ".join(responses)
            final_prompt = f"Combine and summarize these related pieces: {combined_response}"
            
            return self.llm.invoke(final_prompt).content
            
        except Exception as e:
            logger.error(f"Error in safe_invoke: {e}")
            raise


# Main function for testing the ModelManager class
if __name__ == '__main__':
    try:
        model_manager = ModelManager('config/config.yaml')
        model_manager.load_embedding_model()

        # print("Token count:", model_manager.count_tokens("Hello world."))
        print("Groq LLM Response:", model_manager.llm.invoke("Hello world."))
        print("Embedding Result:", model_manager.embedding_model.embed_documents(["Hello world."]))
        # print("LLM content:", model_manager.get_llm_response("Hello world."))
    except Exception as e:
        logger.error("An error occurred: %s", e)