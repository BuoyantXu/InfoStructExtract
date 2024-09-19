from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from settings import batch_key, batch_model, ollama_model

llm_batch = ChatOpenAI(
    temperature=0,
    model=batch_model,
    openai_api_key=batch_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

llm_batch_ollama = ChatOllama(
    temperature=0,
    model=ollama_model,
)
