from langchain_openai import ChatOpenAI

from settings import batch_key, batch_model

llm_batch = ChatOpenAI(
    temperature=0,
    model=batch_model,
    openai_api_key=batch_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
