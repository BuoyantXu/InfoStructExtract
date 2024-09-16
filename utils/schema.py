import json


class Field:
    def __init__(self, id: str, description: str = None, examples: list = None):
        self.id = id
        self.description = description
        self.examples = examples

    def __str__(self):
        return f"{self.id}: {self.description}"


class Date(Field):
    def __init__(self, id: str, description: str = None, examples: list = None):
        super().__init__(id, description, examples)


class Number(Field):
    def __init__(self, id: str, description: str = None, examples: list = None, unit: bool = False):
        super().__init__(id, description, examples)

        self.unit = unit


class Text(Field):
    def __init__(self, id: str, description: str = None, examples: list = None):
        super().__init__(id, description, examples)


class Object:
    def __init__(self, fields: list[Text | Number], prompt_system: str = None, description: str = None,
                 complete_example: str | dict = None,
                 mode: str = "json"):
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
