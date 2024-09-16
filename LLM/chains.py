from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from LLM.models import llm
from prompts import prompt_system_extractor, prompt_user_extractor
from utils.schema import Object


def create_extractor_chain(scheme: Object = None, default_prompt_user: str = prompt_user_extractor,
                           default_prompt_system: str = prompt_system_extractor):
    prompt_system = default_prompt_user if not scheme else scheme.prompt_system
    prompt_user = default_prompt_system if not scheme else scheme.prompt_user
    prompt_user_extractor = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(prompt_system),
            HumanMessagePromptTemplate.from_template(prompt_user)
        ]
    )
    chain_extractor = prompt_user_extractor | llm

    return chain_extractor
