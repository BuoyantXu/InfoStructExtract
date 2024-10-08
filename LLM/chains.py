from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from LLM.models import llm_batch, llm_batch_ollama
from schema.prompts import prompt_system_extractor, prompt_user_extractor
from schema.schema import Object


def create_extractor_chain(scheme: Object = None, llm="zhipu",
                           prompt_user: str = prompt_user_extractor,
                           prompt_system: str = prompt_system_extractor):
    prompt_system = prompt_system if not scheme else scheme.prompt_system
    prompt_user = prompt_user if not scheme else scheme.prompt_user

    prompt_user_extractor = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(prompt_system),
            HumanMessagePromptTemplate.from_template(prompt_user)
        ]
    )
    if llm == "zhipu":
        chain_extractor = prompt_user_extractor | llm_batch
    elif llm == "ollama":
        chain_extractor = prompt_user_extractor | llm_batch_ollama
    else:
        raise ValueError("Invalid llm model")

    return chain_extractor
