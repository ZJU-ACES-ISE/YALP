import re
from abc import ABC, abstractmethod


class Template(ABC):

    def __init__(self, template_text: str, template_id: str = None, sample_limit: int = None):
        super().__init__()
        self.template_text: str = template_text
        self.template_id: str = template_id
        self.sample_limit: int = sample_limit
    #
    # def match_text(self, text) -> bool:
    #     template_regex = self.get_template_regex()
    #     return bool(re.fullmatch(template_regex, text))
    #
    # def match_template(self, template: 'Template') -> bool:
    #     template_example = template.get_template_example()
    #     return self.match_text(template_example)
    #
    # @abstractmethod
    # def get_template_regex(self) -> str:
    #     pass
    #
    # @abstractmethod
    # def get_template_example(self) -> str:
    #     pass
