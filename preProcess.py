import regex as re
import pandas as pd


def load_data_to_dict(log_line, log_format):
    """
        根据 输入的日志模板 解析,headers 可以作为输出表格的表头，log_format 正则表达式解析日志格式
        headers: ['Date', 'Time', 'Pid', 'Level', 'Component', 'Content']
        log_format: re.compile('^(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Pid>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?):\\s+(?P<Content>.*?)$')
    """
    headers, regex = generate_logformat_regex(log_format)
    log_dict = log_to_dict(log_line, regex)
    return log_dict




def generate_logformat_regex(log_format):
    """
        根据 输入的日志模板 解析,headers 可以作为输出表格的表头，regex用于 正则表达式解析日志格式
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)  # 将日志模板 根据<> 拆分为若干个部分
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])  # 将空格转换为 容易空白符号"\s",
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')  # header 去除 "<>"
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dict(log_line, regex):
    """
        解析日志行，返回字典
    """
    match = regex.search(log_line)
    result = {}
    if match:
        result = {k: v for k, v in match.groupdict().items()}
    return result


def loads_data_to_dataframe(log_lines, log_format):
    """
        根据 输入的日志模板 解析,headers 可以作为输出表格的表头，regex用于 正则表达式解析日志格式
        headers: ['Date', 'Time', 'Pid', 'Level', 'Component', 'Content']
        regex: re.compile('^(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Pid>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?):\\s+(?P<Content>.*?)$')
    """
    headers, regex = generate_logformat_regex(log_format)
    df_log = logs_to_dataframe(log_lines, regex, headers)
    return df_log
def logs_to_dataframe(log_lines, regex, headers):
    """
        解析日志行，返回 dataframe
    """
    log_messages = []
    linecount = 0  # 行数
    for line in log_lines:  # 读取每一行
        match = regex.search(line)
        if match:
            message = [match.group(header) for header in headers]  # 根据<?p> 对应header 保存为列表
            log_messages.append(message)
            linecount += 1
    logdf = pd.DataFrame(log_messages, columns=headers)
    # 插入一列 lineId，内容为1至length
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf