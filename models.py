from langchain_openai import ChatOpenAI

from settings import batch_key

llm = ChatOpenAI(
    temperature=0,
    model="glm-4-flash",
    openai_api_key=batch_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
