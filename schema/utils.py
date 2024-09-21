import json
import re

import numpy as np

from schema.schema import Object, Text, Number, Date


def format_json_response(json_str: str, schema: Object = None) -> dict | None:
    try:
        json_pattern = r'```json\n(.*?)\n```'
        json_str = re.search(json_pattern, json_str, re.DOTALL).group(1)
        result = json.loads(json_str)

        if not schema:
            return result

        formatted = {}
        for field in schema.fields:
            formatted[field.id] = format_by_field(field, result)
        return formatted

    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e}")
        return None
    except Exception as e:
        print(f"JSON解码错误: {e}")
        return None


def format_by_field(field: Text | Number, result: dict) -> str | float | np.datetime64 | None:
    field_id = field.id
    if field_id in result.keys():
        # Format by field type
        # (1) Text
        if isinstance(field, Text):
            return result[field_id]

        # (2) Number
        elif isinstance(field, Number):
            if field.unit:
                return number_unit_paser(result[field_id])
            else:
                return result[field_id]

        # (3) Date
        elif isinstance(field, Date):
            if field.date_format:
                return np.datetime64(result[field_id], field.date_format)
            else:
                return result[field_id]
    else:
        return None


def number_unit_paser(number: str) -> float | None:
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
            return float(match_wan.group(1)) * 10 ** 4
        elif match_yi:
            return float(match_yi.group(1)) * 10 ** 8
        else:
            raise ValueError(f"无法识别的数值格式: {number}")
    else:
        return None
