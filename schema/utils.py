import json
import re

import cn2an
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
            if field.keep:
                formatted[field.id + "_raw"] = result.get(field.id)
        return formatted

    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None


def format_by_field(field: Text | Number, result: dict) -> str | float | np.datetime64 | None:
    response_str = result.get(field.id)
    if response_str:
        # Format by field type
        # (1) Text
        if isinstance(field, Text):
            return response_str

        # (2) Number
        elif isinstance(field, Number):
            if field.unit:
                return number_unit_paser(response_str)
            else:
                return response_str

        # (3) Date
        elif isinstance(field, Date):
            if field.date_format:
                return np.datetime64(response_str, field.date_format)
            else:
                return response_str
    else:
        return None


def number_unit_paser(number: str) -> float | None:
    str_number = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '壹', '贰', '叁', '肆', '伍', '陆', '柒',
                  '捌', '玖', '拾', '佰', '仟', '万', '亿', '千', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.']
    try:
        number = ''.join([char for char in number if char in str_number])
    except:
        return None

    if number:
        if re.search(r'\d', number):
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
                number = float(match_non.group(1))
                if number > 10000:
                    number = number / 10000
            elif match_wan:
                number = float(match_wan.group(1))
            elif match_yi:
                number = float(match_yi.group(1)) * (10 ** 8)
            else:
                raise ValueError(f"无法识别的数值格式: {number}")

            return number
        else:
            try:
                number = "".join([char for char in number if
                                  char in ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '壹', '贰', '叁',
                                           '肆', '伍', '陆', '柒', '捌', '玖', '拾', '佰', '仟', '万', '亿', '千']])
                number = cn2an.cn2an(number)
                return number / 10000
            except:
                return None
    else:
        return None
