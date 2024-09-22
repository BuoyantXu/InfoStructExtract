import pandas as pd

from LLM.chains import create_extractor_chain
from batch.batch_steps import step_create_batches, step_upload_batches, step_download_output, step_merge_output, \
    remove_batch_files
from schema.schema import Object, Text, Number

if __name__ == "__main__":
    prompt_system = "你擅长从文本中提取关键信息，精确、数据驱动，重点突出关键信息，根据用户提供的文本片段提取并识别关键数据和事实，将提取并识别到的信息以清晰的 JSON 格式呈现。"

    descriptions = """
# Role: 文本提取识别专家

## Goals
- 从以下合同文本中提取采购编号或项目编号、预算金额、商品明细、采购人名称、采购人地址。如果没有找到返回空字符串，不要返回其他内容。
- 根据提取到的采购人名称和采购人地址，识别采购人所处省级、地级、县级三级行政区划。

## Constrains
- 必须提取合同中所有可见的文本信息。
- 提供每个字段及其对应的内容。
- 确保提取的信息准确且易于识别。
- 根据提取到的采购人名称、地址识别采购人所处行政区划信息。  

## Skills
- 专业的文本提取能力
- 理解并解析合同内容
- 提供准确的字段和内容提取
- 专业的行政区划识别能力

## Workflow
1. 读取并理解给定的文本内容。
2. 提取合同文本中所有可见的文本信息。
3. 确定每个字段及其对应的内容。
4. 输出提取的字段及其内容。
5. 输出识别到的完整行政区划。
"""

    # create schema
    schema = Object(
        prompt_system=prompt_system, description=descriptions,
        fields=[
            Text("项目编号", "项目编号，确定特定的项目。", ["XFZC2018-015", "包采谈〔2018〕1096号"]),
            Number("预算金额", "预算金额，单位为万元或元。", ["350.5万元", "386192.5元"],
                   unit=True, keep=True),
            Text("商品明细", "商品明细，包括名称、数量和单价。"),
            Text("采购人名称", "采购人名称，采购人身份或工作单位",
                 ["澄迈县加乐镇人民政府", "同济大学", "滨州市沾化区文化体育新闻出版局"]),
            Text("采购人地址", "采购人所在地理位置或行政区划", ["昆区三八路24号", "辽宁省大连市高新园区凌工路2号"]),
            Text("采购人行政区划", "采购人所处省级、地级、县级三级行政区划")
        ],
        complete_example={
            "项目编号": "包采谈〔2018〕1096号",
            "预算金额": "238,000.00元",
            "商品明细": [
                {"名称": "商品1", "数量": "2", "单价": "200元"}
            ],
            "采购人名称": "澄迈县加乐镇人民政府",
            "采购人地址": "辽宁省大连市高新园区凌工路2号",
            "采购人行政区划": [
                {"省级": "江西省", "地级": "上饶市", "县级": "广丰区"}
            ]
        },
        mode="json"
    )
    print(schema.prompt_user)

    # Example 1: batch API processing
    df = pd.read_pickle(r"examples/example_data.pkl")

    # step 1. create batches
    # create .jsonl files in "batch/batch_input"
    step_create_batches(df['文本'], schema=schema)

    # step 2. upload batches
    step_upload_batches()

    # step 3. download batches
    step_download_output()

    # step 4. merge output
    step_merge_output(schema=schema)

    # reset the batch files
    remove_batch_files()

    # Example 2: single API call
    df = pd.read_pickle(r"examples/example_data.pkl")
    text = df['文本'][0]

    chain_extractor = create_extractor_chain()
    response = chain_extractor.invoke(text)
    print(response)
