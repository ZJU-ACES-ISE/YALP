import re
from typing import List


def check_template_word(template):
    """
    检验 模板中是否存在单词（字母）
    """
    return bool(re.search(r'[a-zA-Z]', template))


def check_word(text: str, wildcard: str = '<*>') -> bool:
    """
    检验 字符串中是否存在单词（字母）
    """
    text = text.replace(wildcard, '')
    return bool(re.search(r'[a-zA-Z]', text))


def replace_content(log_content: str, regexes: List, wildcard: str = '<<preprocessed>>') -> str:
    """
    规则替换，并且保证最后还需要有单词
    """
    for regex_pattern in regexes:
        processed_content = re.sub(regex_pattern, wildcard, log_content)
        log_content = processed_content if check_word(processed_content, wildcard=wildcard) else log_content  # 验证 规则替换后还有单词
    return log_content


def unique_wildcard(log_template: str, wildcard: str = '<*>') -> str:
    """"将重复的通配符替换为一个"""
    escaped_wildcard = re.escape(wildcard)
    processed_template = re.sub(fr'([\w.-]+)?{escaped_wildcard}([\w.-]+)?', wildcard, log_template)  # 通配符合并左右单词部分
    log_template = processed_template if check_word(processed_template, wildcard=wildcard) else log_template  # 验证 规则替换后还有单词
    log_template = re.sub(fr"({escaped_wildcard})([\s.\-\\/]*{escaped_wildcard})*", wildcard, log_template)  # 通配符合并相邻通配符
    return log_template


def check_one_word(log_content: str) -> bool:
    """
    验证是否仅一个单词，或者没有单词
    """
    return len(re.findall('(\w+)', log_content)) < 2


def check_match(log_template: str, log_content: str, wildcard='<*>') -> bool:
    """
    验证 模板是否匹配 日志
    """
    log_template = re.escape(log_template)
    pattern = re.sub(r"(<\\\*>(\\\s)*)+", ".*?", log_template)
    log_content.replace(wildcard, '\1')
    return bool(re.fullmatch(pattern, log_content))


def check_match_all(log_template, log_messages):
    """
    验证 模板是否匹配 使用日志
    """
    for log_message in log_messages:
        if not check_match(log_template, log_message):
            return False
    return True


def isword(str, wildcard='<*>'):
    """
    检验 字符串中是否存在单词（字母）或者通配符
    """
    return bool(re.search(r'[a-zA-Z]', str)) or str == wildcard


def special_character_replace(text):
    special_characters = {'<': '＜', '>': '＞', '{': '｛', '}': '｝'}
    for k, v in special_characters.items():
        text = text.replace(k, v)
    return text


def special_character_restore(text):
    text = text.replace('{V}', '')

    special_characters = {'<': '＜', '>': '＞', '{': '｛', '}': '｝'}
    for k, v in special_characters.items():
        text = text.replace(v, k)
    return text
