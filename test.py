from LLM.chains import create_extractor_chain
from utils.schema import Object, Text, Number

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

schema = Object(
    prompt_system="你擅长从文本中提取关键信息，精确、数据驱动，重点突出关键信息，根据用户提供的文本片段提取关键数据和事实，将提取的信息以清晰的 JSON 格式呈现。",
    description=descriptions,
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
print(schema.prompt_system)
print(schema.prompt_user)

chain_extractor = create_extractor_chain(schema)
r = chain_extractor.invoke("""
包头市昆都仑区环境卫生综合服务中心融雪剂采购采购公告
项目名称：	包头市昆都仑区环境卫生综合服务中心融雪剂采购
项目编号：	包采谈〔2018〕1096号
采购代理机构内部编号：	XFZC2018-015
采购目录：	货物类\基础化学品及相关产品
采购方式：	竞争性谈判
评标方式：	纸质评标
供应商投标资格：	1、在中华人民共和国注册且具有独立法人资格的生产企业； 2、投标单位须在包头市政府采购网“投标单位注册”中“申报注册”填写“包头市政府采购投标单位准入申请登记表”注册审核通过，查询状态为“有效”。 3、具有良好的商业信誉和健全的财务会计制度； 4、具有依法纳税和缴纳社会保障资金的良好记录； 5、参加本次采购活动前三年内，在经营活动中没有重大违法记录； 6、具备国家CMA认证机构并且符合本次招标内容的2018年9月1日之后出具的融雪剂检测报告（固体溶解速度、冰点、pH值、碳钢腐蚀率、路面摩擦衰减率、植物种子相对受害率等）； 7、近三年（2016年起至今）至少完成3项融雪剂业绩； 8、在国家企业信用信息公示系统（http://www.gsxt.gov.cn/）中未被列入严重违法失信企业名单的证明材料； 9、供应商在“信用中国”网站（http：//www.creditchina.gov.cn/）中未被列入失信被执行人名单； 10、本项目不接受联合体参与谈判； 11、单位负责人为同一人的供应商，不得同时参加本项目的谈判；
用途、数量和简要技术要求：	详见招标文件
获取采购文件开始时间：	2018年11月26日 09:00
获取采购文件截止时间：	2018年11月30日 17:00
工作时间	上午9:00~12:00   下午14:30~17:00
获取文件地址：	包头市昆都仑区苏宁广场苏宁雅悦公寓A座913
采购文件售价(元)：	500.00
答疑会时间：	 
答疑地址：	 
投标文件递交截止时间：	2018年12月04日 14:00
文件递交地址：	包头市公共资源交易中心三楼开标室
开标时间：	2018年12月04日 14:00
开标地址：	包头市公共资源交易中心三楼开标室
采购人名称：	包头市昆都仑区环境卫生综合服务中心
采购人地址：	昆区三八路24号
采购人联系人：	张惠明
采购人联系方式：	13848253689
采购代理机构名称：	包头市鑫丰项目管理有限公司
采购代理机构地址：	包头市昆都仑区青年路15号街坊1-031-502
采购代理机构网址：	 
采购代理机构银行帐号：	15050171666400000120
采购代理机构开户行：	中国建设银行股份有限公司包头建业支行
行政主管部门：	昆区政府采购管理办公室
项目负责人：	杨烨、刘金豆
代理机构联系电话：	0472-6982211
采购文件：	 采购公告.docx
项目预算合计:
包号	包组预算合计
A	238,000.00
采购内容为：
采购条目流水号	设备名称	单位	数量	采购明细清单
9097658	货物	 	1.0	采购融雪剂
质疑方式：依据《政府采购法》第五十二条、《政府采购实施条例》第五十三条。供应商认为招标文件存在倾向性、歧视性条款，损害其合法权益的，可以在获取招标文件之日起7个工作日内，且在投标截止之日前，以书面形式向包头市昆都仑区环境卫生综合服务中心、包头市鑫丰项目管理有限公司提出质疑，逾期不予受理。供应商对质疑答复不满意的，或者采购人、采购代理机构未在规定期限内作出答复的，可以在质疑答复期满后15个工作日内，向同级财政部门提出投诉，逾期不予受理。
""")
print(r.content)