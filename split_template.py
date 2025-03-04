from typing import List, Tuple, Dict

from Sample import Sample
from Template import Template
from utils.utils_template import find_sequence_indices, adjust_template, get_lcs_of_tokenized_texts, \
    tokenize_text_with_punctuation_regex
from utils.utils_match import unique_wildcard, check_match, check_word


def get_private_part(com_part, tokenized_samples: List[List[str]]):
    """
        得到公共部分的私有部分
    """
    samples_part = [find_sequence_indices(sample_tokenized, com_part) for sample_tokenized in tokenized_samples]

    private_dicts = []
    for i in range(len(com_part) + 1):
        part_index = i * 2
        private_dict = dict()
        for sample_index in range(len(samples_part)):
            part = samples_part[sample_index][part_index]
            if part in private_dict:
                private_dict[part].append(sample_index)
            else:
                private_dict[part] = [sample_index]
        private_dicts.append(private_dict)
    return private_dicts


def merge_include_part(part_dict: Dict) -> Dict:
    """ 私有部分 合并"""
    results = part_dict.copy()
    for template, samples in results.items():
        # 检查其他键是否包含当前值
        for other_template, other_samples in results.items():
            # 如果 value 可以 覆盖 other_value 或者 other_key
            if other_template != template and check_match(template, other_template):
                results[template].extend(other_samples)
                results[other_template] = []
    return {unique_wildcard(adjust_template(key)): value for key, value in results.items() if len(value) > 0}


def merge_include_template(templates: List[Tuple[Template, List[Sample]]]) -> List[Tuple[Template, List[Sample]]]:
    """ 模板合并"""
    results = templates.copy()
    for template, samples in results:
        # 检查其他键是否包含当前值
        for other_template, other_samples in results:
            # 如果 value 可以 覆盖 other_value 或者 other_key
            if other_template.template_text != template.template_text and check_match(template.template_text, other_template.template_text):
                samples.extend(other_samples)
                other_samples.clear()
    return [(key, value) for key, value in results if len(value) > 0]


def get_score(private_dict, inf):
    """ 处理私有部分的有效性得分"""
    score = len(private_dict)
    if len(private_dict) == 1:
        score = inf
    else:
        word_flag = False
        for p in private_dict.keys():
            if check_word(p):
                word_flag = True
                break
        if word_flag == False:
            score = inf
    return score


def get_min_private(private_dicts, limit, len_sample):
    """ 获取最小的私有部分"""
    scores = [get_score(private_dict, len_sample + 1) for private_dict in private_dicts]
    min_index = min(enumerate(scores), key=lambda x: x[1])[0]

    if scores[min_index] > limit:
        min_index = -1  # 无效化处理
        min_private = {'': range(len_sample)}
    else:
        min_private = private_dicts[min_index]

    return min_index, min_private


def check_result_word(results):
    """检查是否有单词，如果没有单词的话，使用样本作为模版"""
    new_result = []
    for template, samples in results:
        if check_word(template.template_text):
            new_result.append((template, samples))
        else:
            for sample in samples:
                new_result.append((Template(sample.sample_text), [sample]))
    return new_result


def split_template_by_lcs(samples: List[Sample], limit: int) -> List[Tuple[Template, List[Sample]]]:
    """
        模板拆分，基于lcs方法实现
    """
    tokenized_samples = [tokenize_text_with_punctuation_regex(sample.sample_text) for sample in samples]  # 分词
    com_part = get_lcs_of_tokenized_texts(tokenized_samples)  # 最长公共子序列
    private_dicts = get_private_part(com_part, tokenized_samples)  # 获取私有部分
    private_dicts = [merge_include_part(private_dict) for private_dict in private_dicts]  # 私有部分检测覆盖问题
    min_index, min_private = get_min_private(private_dicts, limit, len(samples))  # 获取最小的私有部分
    split_results = []

    com_part.append([''])
    com_part = [''.join(sub_part) for sub_part in com_part]
    for part, sample_indices in min_private.items():  # 遍历最小的私有部分
        if part == '' and min_index != 0 and min_index != len(com_part):
            part = '{V}'
        template_text = ''
        for i in range(len(com_part)):  # 拼接模板
            template_text += part if i == min_index else list(private_dicts[i].keys())[0] if len(private_dicts[i]) == 1 else '<*>'
            template_text += com_part[i]

        new_samples = [samples[sample_index] for sample_index in sample_indices]
        template_text = unique_wildcard(template_text)
        new_template = Template(template_text)

        split_results.append((new_template, new_samples))

    split_results = check_result_word(split_results)  # 检查是否剩余单词
    split_results = merge_include_template(split_results)  # 检查覆盖问题

    for new_template, new_samples in split_results:  # 更新sample_limit
        new_log_len = sum([sample.log_num for sample in new_samples])
        new_template.sample_limit = new_log_len * 2

    return split_results
