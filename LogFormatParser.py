import re


class LogFormatParser():
    def __init__(self):
        self.headers = None
        self.regex = None

    def generate_logformat_regex(self, log_format):
        self.headers = []
        splitters = re.split(r'(<[^<>]+>)', log_format)  # 将日志模板 根据<> 拆分为若干个部分
        self.regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])  # 将空格转换为 容易空白符号"\s",
                self.regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')  # header 去除 "<>"
                self.regex += '(?P<%s>.*?)' % header
                self.headers.append(header)
        self.regex = re.compile('^' + self.regex + '$')

    def log_to_dict(self, log_line):
        match = self.regex.search(log_line)
        result = {}
        if match:
            result = {k: v for k, v in match.groupdict().items()}
        return result
