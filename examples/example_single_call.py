import pandas as pd
from tqdm import tqdm

from LLM.chains import create_extractor_chain
from schema.schema import Object, Text, Number
from schema.utils import format_json_response

if __name__ == "__main__":
    # Example 1: single call
    # set system prompt and descriptions for goals, constrains, skills and workflow
    prompt_system = "你擅长从文本中提取关键信息，精确、数据驱动，重点突出关键信息，根据用户提供的文本片段提取关键数据和事实，将提取的信息以清晰的 JSON 格式呈现。"
    descriptions = """
    # Role: 文本提取专家
    
    ## Goals
    - 从以下合同文本中提取采购编号或项目编号、预算金额。如果没有找到返回空字符串，不要返回其他内容。
    
    ## Constrains
    - 必须提取合同中所有可见的文本信息。
    - 提供每个字段及其对应的内容。
    - 确保提取的信息准确且易于识别。
    
    ## Skills
    - 专业的文本提取能力
    - 理解并解析合同内容
    - 提供准确的字段和内容提取
    
    ## Workflow
    1. 读取并理解给定的文本内容。
    2. 提取合同文本中所有可见的文本信息。
    3. 确定每个字段及其对应的内容。
    4. 输出提取的字段及其内容。
    """

    # create schema
    schema = Object(
        prompt_system=prompt_system, description=descriptions,
        fields=[
            Text("项目编号", "项目编号，确定特定的项目。", ["XFZC2018-015", "包采谈〔2018〕1096号"]),
            Number("预算金额", "预算金额，单位为万元或元。", ["350.5万元", "386192.5元"], unit=True),
        ],
        complete_example={
            "项目编号": "包采谈〔2018〕1096号",
            "预算金额": "23.5万元",
        },
        mode="json"
    )

    # print formatted system and user prompts
    print(schema.prompt_system)
    print(schema.prompt_user)

    # Single call to extract information from text
    # create extractor chain
    chain_extractor = create_extractor_chain(schema, llm="ollama")
    df = pd.read_pickle("batch/batch_chunks/page_text_chunk_1.pkl")

    text_list = df['文本'].tolist()
    results = [format_json_response(json_str=chain_extractor.invoke(text).content, schema=schema) for text in
               tqdm(text_list[70:100])]
    r = chain_extractor.invoke(df['文本'][0])
    # print response
    print(r.content)

    # format response into dict with schema
    result = format_json_response(json_str=r.content, schema=schema)
    print(result)
