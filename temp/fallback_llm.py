from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler


gemini_llm = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0
)
ollama_llm = init_chat_model("ollama:deepseek-r1:7b", reasoning=False)
print(gemini_llm)
print(ollama_llm)
llm = gemini_llm.with_fallbacks([ollama_llm])
usage_callback = UsageMetadataCallbackHandler()
print(llm.invoke("Hi", config={"callbacks": [usage_callback]}))
print(usage_callback)
