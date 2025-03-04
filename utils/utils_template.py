import re
from typing import List
import numpy as np
from sklearn.cluster import MeanShift, DBSCAN

from Template import Template
from utils.utils_match import isword, check_match


def prune_template_by_distance(templates: List[Template], new_template: Template, limit: float = 0.5):
    """
    对模板列表进行剪枝，去除与新模板的编辑距离大于limit的模板
    """
    templates = [template for template in templates if lcs_distance(template, new_template) <= limit]
    return templates


def get_matrix_of_distance(data: List[Template]) -> np.ndarray:
    """计算编辑距离矩阵"""
    num_strings = len(data)
    dist_matrix = np.zeros((num_strings, num_strings), dtype=np.float32)
    for i in range(num_strings):
        for j in range(i + 1, num_strings):  # 只计算上半部分
            dist = lcs_distance(data[i], data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # 填充对称的位置
    return dist_matrix


def get_cluster(dist_matrix):
    """聚类"""

    def get_cluster_with_meanshift(dist_matrix):
        cluster = MeanShift(bandwidth=None)  # MeanShift聚类
        cluster.fit(dist_matrix)
        return cluster.labels_

    def get_cluster_dbscan(dist_matrix):
        dbscan = DBSCAN(eps=0.3, min_samples=1)
        dbscan.fit(dist_matrix)
        return dbscan.labels_

    return get_cluster_dbscan(dist_matrix)


def lcs_distance(template1: Template, template2: Template) -> int:
    """
    计算两个模板的编辑距离
    """
    tokenized_t1 = tokenize_text_with_punctuation_regex(template1.template_text)
    t1_tokens = sum(map(isword, tokenized_t1))
    tokenized_t2 = tokenize_text_with_punctuation_regex(template2.template_text)
    t2_tokens = sum(map(isword, tokenized_t2))
    com_part = all_common_substrings(tokenized_t1, tokenized_t2)
    com_tokens = sum([sum(map(isword, sub_com)) for sub_com in com_part])

    max_x_tokens = max(t1_tokens, t2_tokens)
    if max_x_tokens == 0:
        return 1
    else:
        return 1 - com_tokens / max_x_tokens


def com_part_to_tokenized(com_part, delimiter="<<COM>>"):
    """将公共部分块进行分词化"""  # 因为 计算LCS 需要的是列表，不能是序列快
    result = [delimiter]
    for sub_part in com_part:
        result.extend(sub_part)
        result.append(delimiter)
    return result


def all_common_substrings(text1, text2):
    """
    计算两个文本的所有公共子序列块
    """
    m, n = len(text1), len(text2)

    # 创建一个二维数组用于存储中间结果
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充dp数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 通过dp数组回溯得到最长公共子序列
    lcs_list = []
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        else:
            if dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
            if len(lcs) > 0:
                lcs_list.append(lcs[::-1])
            lcs = []
    if len(lcs) > 0:
        lcs_list.append(lcs[::-1])
    return lcs_list[::-1]


def all_common_substrings_power(text1, text2):
    """
    计算两个文本的所有公共子序列块
    """
    # 计算 最长公共子序列块
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建一个二维数组用于存储中间结果
    for i in range(1, m + 1):  # 填充dp数组
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                com = text1[i - 1].replace(r"<*>", 'wildcard')
                tokens = len(com.split())

                non_word_space_match = re.findall(r'[^\w\s]+', com)
                non_word_space_length = sum(len(part) for part in non_word_space_match)
                dp[i][j] = dp[i - 1][j - 1] + tokens * 3 + max(non_word_space_length // 3, 1)

            else:  # 纯字符
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # 通过dp数组回溯得到最长公共子序列
    lcs_list = []
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        else:
            if dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
            if len(lcs) > 0:
                lcs_list.append(lcs[::-1])
            lcs = []
    if len(lcs) > 0:
        lcs_list.append(lcs[::-1])
    return lcs_list[::-1]


def tokenize_text_with_punctuation_regex(text):
    """分词,使用正则表达式将标点符号和单词一起分隔"""
    tokens = re.findall(r'[\w\-\\/.]+|\(?<\*>\)?|\W', text)
    return tokens


def get_lcs_of_tokenized_texts(tokenized_texts: List[List[str]]) -> List[List[str]]:
    """
    计算多个文本的最长公共子序列块
    """
    if len(tokenized_texts) == 1:  # 不应该出现为空的情况
        return [tokenized_texts[0]]
    com_part = all_common_substrings_power(tokenized_texts[0], tokenized_texts[1])
    for sample_tokenized in tokenized_texts[2:]:
        com_part = com_part_to_tokenized(com_part)
        com_part = all_common_substrings_power(com_part, sample_tokenized)
    return com_part


def adjust_template(origin_template):
    """
    调整模板，将 <<*>> 替换为 <*>
    """
    pattern = r"(<<(.*?)>>)+|(<(.*?)>)+"
    res = re.sub(pattern, r"<*>", origin_template)
    return res


def merge_template(templates):
    """
    模板合并
    """
    for index, template in enumerate(templates):
        # 检查其他键是否包含当前值
        for other_index, other_template in enumerate(templates):
            # 如果 value 可以 覆盖 other_value 或者 other_key
            if check_match(template, other_template):
                templates[other_index] = template

    return templates


def unique_merge_template(templates):
    """
    去重，合并，去重
    """
    templates = np.unique(templates)
    templates = merge_template(templates)
    results = np.unique(templates)
    return results


def find_subarray_indices(lst, subarray, start_index):
    """
    在给定的列表中查找子数组的起始索引
    """
    sub_len = len(subarray)
    for i in range(start_index, len(lst) - sub_len + 1):
        if lst[i:i + sub_len] == subarray:
            return i


def find_sequence_indices(lst, sequence):
    """
    将给定的列表，根据序列类比列表，拆分为子数组序列
    """
    indices = []
    start_index = 0
    for item in sequence:
        index = find_subarray_indices(lst, item, start_index)
        indices.append(''.join(lst[start_index:index]))
        indices.append(''.join(item))
        start_index = index + len(item)
    indices.append(''.join(lst[start_index:]))

    return indices


def tokenize_text_with_punctuation(text):
    """分词，使用正则表达式将标点符号和单词一起分隔"""
    tokens = re.findall(r'\w+|<\*>|\W', text)
    return tokens


def fix_template(log_content, log_template):
    """
    修复模板
    """
    log_content = adjust_template(log_content)
    lst_c = tokenize_text_with_punctuation(log_content)
    lst_t = tokenize_text_with_punctuation(log_template)
    com_list = all_common_substrings(lst_c, lst_t)
    result_c = find_sequence_indices(lst_c, com_list)
    result_t = find_sequence_indices(lst_t, com_list)

    for i in range(0, len(result_t)):  # result_c，result_t 的长度是 len(com_list)*2+1
        if result_c[i] != result_t[i] and '<*>' in result_t[i]:
            result_c[i] = '<*>'
    result_fixed = ''.join(result_c)
    return result_fixed
