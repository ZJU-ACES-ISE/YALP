import re
import difflib
import pandas as pd
from collections import Counter


# 处理和清洗单个模板
def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean)
    US (User String)
    DG (Digit)
    PS (Path-like String)
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """
    # 定义一些常量，其中default_strings是指用户自定义的字符串，是要根据数据集修改的，这里使用的是研究者对数据集总结出来的
    boolean = {'true', 'false'}
    default_strings = {'null', 'root', 'admin'}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:',
        r'\=', r'\|', r'\"', r'\'',
        r'\[', r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    # 如果提供了用户字符串，将其添加到默认字符串中
    if user_strings:
        default_strings = default_strings.union(user_strings)

    # DS：删除多余的空格（匹配一个或多个连续的空白字符替换为一个空格）
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # PS：类似路径的变量变成<*>
    # 将path_delimiters中的分符隔拼接成一个正则表达式，并将其作为分隔符进行分割，存在p_tokens
    p_tokens = re.split('(' + '|'.join(path_delimiters) + ')', template)
    new_p_tokens = []

    # 判断p_token是否符合路径样式字符串的格式，即以斜杠开，头后面跟着一个或多个非斜杠字符
    for p_token in p_tokens:
        if re.match(r'^(\/[^\/]+)+$', p_token):
            p_token = '<*>'
        new_p_tokens.append(p_token)

    # print(template)
    template = ''.join(new_p_tokens)
    # print("后处理", template)

    # 处理剩余的规则
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)  # tokenizing while keeping delimiters
    new_tokens = []
    # print(tokens)
    for token in tokens:
        # BL, US:布尔型和用户自定义型
        for to_replace in boolean.union(default_strings):
            if token.lower() == to_replace.lower():
                # print(token)
                token = '<*>'

        # # 数字类型
        # if re.match(r'^\d+$', token):
        #     token = '<*>'
        #
        # # WV：检查 token 是否符合 <*><*> 的格式，两个<*>之间不包含空格或/
        # if len(tokens) > 1 and re.match(r'^[^\s\/]*<\*>[^\s\/]*$', token):
        #     if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
        #         token = '<*>'

        # collect the result
        new_tokens.append(token)

    # make the template using new_tokens
    template = ''.join(new_tokens)

    # DV：用.分隔的两个变量合并成一个
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # 替换连续的中间没有分隔符的变量
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    # 替换特定的字符串  #<*># 和 <*>:<*>
    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")
    return template


def edit_distance_str(s1, s2):
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return 1.0 - matcher.ratio()


def combine_template():
    return


def post_process(in_structured_path, out_structured_path, out_templates_path):
    param_regex = [
        # r'{([ :_#.\-\w\d]+)}',
        # r'{}'
    ]

    log_df = pd.read_csv(in_structured_path)
    content = log_df.Content.tolist()
    template = log_df.EventTemplate.tolist()
    for i in range(len(content)):
        t = str(template[i])
        for r in param_regex:
            t = re.sub(r, "<*>", t)
        template[i] = correct_single_template(t)
    log_df.EventTemplate = pd.Series(template)
    unique_templates = sorted(Counter(template).items(), key=lambda k: k[1], reverse=True)
    temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])
    log_df.to_csv(out_structured_path)
    temp_df.to_csv(out_templates_path)


if __name__ == '__main__':
    datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
                'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']
    in_dir = f"./datasets/2k_datasets"
    for dataset in datasets:
        in_structured_path = f"{in_dir}/{dataset}/{dataset}_2k.log_structured_corrected.csv"
        out_structured_path = f"{in_dir}/{dataset}/{dataset}_2k.log_structured_corrected_post.csv"
        out_templates_path = f"{in_dir}/{dataset}/{dataset}_2k.log_templates_corrected_post.csv"
        post_process(in_structured_path, out_structured_path, out_templates_path)
