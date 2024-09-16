import json
import re

from utils.schema import Object, Text, Number


def format_json(json_str: str, schema: Object) -> dict:
    try:
        result = json.loads(json_str)
    except:
        raise ValueError(f"无法解析的 JSON 字符串: {json_str}")

    formatted = {}
    for field in schema.fields:
        formatted[field.id] = format_by_field(field, result)
    return formatted


def format_by_field(field: Text | Number, result: dict) -> str | float | None:
    field_id = field.id
    if field_id in result.keys():
        # Format by field type
        # (1) Text
        if isinstance(field, Text):
            return result[field_id]

        # (2) Number
        elif isinstance(field, Number):
            if field.unit:
                return unit_paser(result[field_id])
            else:
                return result[field_id]
    else:
        return None


def unit_paser(number: str) -> float | None:
    if number:
        # 匹配规则
        pattern_non = r'([\d,]+\.?\d*)\s*'
        pattern_wan = r'([\d,]+\.?\d*)\s*万'
        pattern_yi = r'([\d,]+\.?\d*)\s*亿'

        # 去除千位分隔符
        amount_str = number.replace(',', '')

        # 匹配并转换
        match_non = re.search(pattern_non, amount_str)
        match_wan = re.search(pattern_wan, amount_str)
        match_yi = re.search(pattern_yi, amount_str)

        if match_non:
            return float(match_non.group(1))
        elif match_wan:
            return float(match_wan.group(1)) * 10000
        elif match_yi:
            return float(match_yi.group(1)) * 100000000
        else:
            raise ValueError(f"无法识别的数值格式: {number}")
    else:
        return None
