# ==========================================================================
# Master Thesis - LLM Factory
#   - André Filipe Gomes Silvestre, 2025
# 
#   Implements a Factory Pattern to instantiate LLMs. 
#   Supports:
#     - Google (Gemma/Gemini)
#     - OpenAI (GPT models)
#     - Groq (High-speed API for open models like Qwen/Llama)
#     - Ollama (Local execution)
# ==========================================================================

import os
from langchain_core.language_models.chat_models import BaseChatModel
from config import Config

class LLMFactory:
    """
    Factory class to create Language Model instances based on configuration.
    """

    @staticmethod
    def get_llm(provider: str = Config.MODEL_PROVIDER) -> BaseChatModel:
        """
        Returns a configured ChatModel instance.

        Args:
            provider (str): 'google', 'openai', 'groq', or 'ollama'.

        Returns:
            BaseChatModel: An instance of a LangChain chat model.
        
        Raises:
            ValueError: If the provider is not supported or API keys are missing.
        """
        
        # 1. GOOGLE (Gemma / Gemini)
        if provider == "google":
            if not Config.GOOGLE_API_KEY:
                raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")
            
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=Config.GOOGLE_MODEL_NAME,
                temperature=Config.TEMPERATURE,
                google_api_key=Config.GOOGLE_API_KEY,
                convert_system_message_to_human=True # Sometimes needed for older Gemini versions
            )

        # 2. OPENAI (GPT-OSS / GPT-4)
        elif provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")
            
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=Config.OPENAI_MODEL_NAME,
                temperature=Config.TEMPERATURE,
                api_key=Config.OPENAI_API_KEY
            )

        # 3. GROQ (For Qwen via API - Free Tier)
        # Using Groq is the best way to run Qwen/Llama via API without local GPU load.
        elif provider == "groq":
            if not Config.GROQ_API_KEY:
                raise ValueError("❌ GROQ_API_KEY not found. Get one at console.groq.com")
            
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=Config.GROQ_MODEL_NAME,
                temperature=Config.TEMPERATURE,
                api_key=Config.GROQ_API_KEY
            )

        # 4. OLLAMA (Local Fallback)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=Config.LOCAL_MODEL_NAME,
                temperature=Config.TEMPERATURE,
                keep_alive="5m"
            )

        else:
            raise ValueError(f"❌ Unsupported model provider: {provider}")

# ==========================================================================
# Test Block
# ==========================================================================
if __name__ == "__main__":
    # Simple test to verify connectivity
    try:
        print(f"🔄 Initializing LLM with provider: {Config.MODEL_PROVIDER}...")
        llm = LLMFactory.get_llm()
        
        # Determine model name for printing
        model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'Unknown'))
        print(f"✅ LLM Ready: {model_name}")
        
        print("🧪 Sending test prompt...")
        response = llm.invoke("Explain briefly what is the capital of Portugal.")
        print(f"🤖 Response:\n{response.content}")
        
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("💡 Tip: Check if your API Keys are set in .env or environment variables.")