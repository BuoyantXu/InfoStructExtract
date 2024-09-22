import json


class Field:
    """
    Field class to define the fields in the schema.

    :param id: str, the id of the field
    :param description: str, the description of the field
    :param examples: list, the examples of the field
    :param keep: bool, if True it will keep raw string as {colname}_raw
    """

    def __init__(self, id: str, description: str = None, examples: list = None, keep: bool = False):
        self.id = id
        self.description = description
        self.examples = examples
        self.keep = keep

    def __str__(self):
        return f"{self.id}: {self.description}"


class Date(Field):
    """
    Date field class to define the date field in the schema.

    :param date_format: str, the format of the date
    """

    def __init__(self, id: str, description: str = None, examples: list = None, keep: bool = False, date_format: str = "YYYY-MM-DD"):
        super().__init__(id, description, examples, keep)

        self.date_format = date_format


class Number(Field):
    """
    Number field class to define the number field in the schema.

    :param unit: bool, whether the number has a unit
    """

    def __init__(self, id: str, description: str = None, examples: list = None, keep: bool = False, unit: bool = False):
        super().__init__(id, description, examples, keep)

        self.unit = unit


class Text(Field):
    """
    Text field class to define the text field in the schema.
    """

    def __init__(self, id: str, description: str = None, examples: list = None, keep: bool = False):
        super().__init__(id, description, examples, keep)


class Object:
    """
    Object class to define the schema object. Create system and user prompts based on the schema.

    :param fields: list, the fields in the schema
    :param prompt_system: str, the system prompt
    :param description: str, the description of the schema
    :param complete_example: str | dict, the complete example of the schema
    :param mode: str, the mode of the schema which can be "json" or "yaml"
    """

    def __init__(self, fields: list[Text | Number], prompt_system: str = None, description: str = None,
                 complete_example: str | dict = None, mode: str = "json"):
        self.prompt_system = prompt_system
        self.description = description
        if isinstance(complete_example, dict):
            self.complete_example = json.dumps(complete_example, ensure_ascii=False, indent=4)
        else:
            self.complete_example = str(complete_example)
        self.fields = fields
        self.mode = mode

        self.ids = [field.id for field in fields]

        # format prompt
        self.field_description = self.format_field_description()
        self.complete_example = self.format_complete_example()

        self.prompt_user = self.format_prompt_user()

    def format_field_description(self) -> str:
        field_description = "## Field Descriptions\n"
        for field in self.fields:
            if field.examples:
                field_description += f"- **{field.id}**: {field.description} 例如：{', '.join(field.examples)}。如果没有找到，返回空字符串。\n"
            else:
                field_description += f"- **{field.id}**: {field.description} 如果没有找到，返回空字符串。\n"
        return field_description

    def format_complete_example(self) -> str:
        if self.complete_example:
            return f"## Example\n```{self.mode}\n{self.complete_example}\n```"
        else:
            return ""

    def format_prompt_user(self) -> str:
        prompt_user = f"""{self.description}\n{self.field_description}\n{self.complete_example}\n\n文本内容：""".replace(
            "{", "{{").replace("}", "}}")
        prompt_user += "\n'''\n{text}\n'''"
        return prompt_user
