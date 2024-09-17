import pandas as pd
from schema.schema import Object, Text, Number

from batch.batch_steps import step_create_batches, step_upload_batches, step_download_output, step_merge_output, \
    remove_batch_files

if __name__ == "__main__":
    # Example 2: batch API processing
    prompt_system = "你擅长从文本中提取关键信息，精确、数据驱动，重点突出关键信息，根据用户提供的文本片段提取关键数据和事实，将提取的信息以清晰的 JSON 格式呈现。"
    descriptions = """
    # Role: 文本提取专家

    ## Goals
    - 从以下合同文本中提取采购编号或项目编号、预算金额、商品明细。如果没有找到返回空字符串，不要返回其他内容。

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
            Text("商品明细", "商品明细，包括名称、数量和单价。")
        ],
        complete_example={
            "项目编号": "包采谈〔2018〕1096号",
            "预算金额": "238,000.00元",
            "商品明细": [
                {"名称": "商品1", "数量": "2", "单价": "200元"}
            ]
        },
        mode="json"
    )

    # print formatted system and user prompts
    print(schema.prompt_system)
    print(schema.prompt_user)

    df = pd.read_pickle(r"examples/example_data.pkl")

    # step 1. create batches
    # create .jsonl files in "batch/batch_input"
    step_create_batches(df['文本'], schema=schema)

    # step 2. upload batches
    step_upload_batches()

    # step 3. download batches
    step_download_output()

    # step 4. merge output
    step_merge_output()

    # reset the batch files
    remove_batch_files()
