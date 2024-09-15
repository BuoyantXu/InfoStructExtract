from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from models import llm
from prompts import prompt_system_extractor, prompt_user_extractor

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(prompt_system_extractor),
        HumanMessagePromptTemplate.from_template(prompt_user_extractor)
    ]
)
chain_extractor = prompt | llm
