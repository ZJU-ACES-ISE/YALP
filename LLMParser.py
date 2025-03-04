# -*- coding: utf-8 -*-
import re
import time
import traceback

from utils.utils_match import unique_wildcard, check_word, check_match
from utils.utils_template import adjust_template, fix_template

system_message_zero_shot = """    You will be provided with a log message delimited by backticks. You must abstract variables with `<<placeholders>>` to extract the corresponding template.
Print the input log's template delimited by backticks.\nLog message: `{0}`"""

retry_nomatch_message = """The template can not be be matched by the log. """ + \
                        """You must abstract variables to extract the corresponding template with `<<placeholders>>`.""" + \
                        """Just only format the output of the template delimited by backtick, without any explanation."""

retry_noresult_message = """The output is formatted incorrectly; the content contained within backtick is not available. """ + \
                         """Just only format the output of the template delimited by backtick, without any explanation."""


class LLMParser:
    def __init__(self, llm):
        self.llm = llm
        self.error_list, self.error_list_nomatch, self.error_list_noword = [], [], []
        self.total_tokens = 0
        self.max_retries = 3  # 对话异常最大次数
        self.init_batch_size = 10  # 批量处理 的初始批次大小，每次减半
        self.max_message_length = 7000  # gtp对话时，设置最大消息长度
        self.total_time = 0
        self.parse_count = 0  # 解析次数，不包含重试

    def get_completion_loop(self, log_content, messages):
        """循环向gpt对话获取对话内容"""
        retry_count = 0  # 当前重试次数
        self.parse_count += 1
        result = ''
        while retry_count < self.max_retries:  # 最大重试次数
            try:
                time_start = time.time()
                result = self.llm.get_completion_from_messages(messages)
                self.total_time += time.time() - time_start

                result, messages = self.check_retry(
                    result, log_content, messages)
                if messages is None:
                    return result
                else:
                    raise Exception(f"对话异常,\nmessages：{messages}\nresult：{result}")

            except Exception as e:
                retry_count += 1
                traceback.print_exc()
                continue
        return result  # 多次重试依然无法处理,直接返回对话结果

    def correct_result(self, log_content, template_result):
        """对对话结果进行检查纠正"""
        template_adjust = adjust_template(template_result)  # 调整模板
        if not check_match(template_adjust, log_content):  # 检查是否匹配
            self.error_list_nomatch.append(log_content)
            template_adjust = fix_template(log_content, template_adjust)
        if not check_match(template_adjust, log_content):  # 检查是否包含模板词
            self.error_list_noword.append(log_content)
            template_adjust = log_content
        template_adjust = adjust_template(template_adjust)  # 调整模板
        template_adjust = unique_wildcard(template_adjust)  # 唯一通配符

        if not check_word(template_adjust):  # 检查是否包含模板词
            self.error_list_noword.append(log_content)
            template_adjust = log_content
        template_adjust = adjust_template(template_adjust)  # 调整模板
        template_adjust = unique_wildcard(template_adjust)  # 唯一通配符
        return template_adjust

    def process_one_log(self, log_content, messages):
        """处理单条日志"""
        result = self.llm.get_completion_from_messages(messages)
        pattern = r"`+([^`]+)`+"
        matches = re.findall(pattern, result)
        messages.append({'role': 'assistant', 'content': result})
        if len(matches) == 1:
            result = matches[0]

        template_adjust = adjust_template(result)
        if not check_match(template_adjust, log_content):
            result = log_content

        template_adjust = self.correct_result(log_content, result)
        return template_adjust

    def parse_message_zero_shot(self, log_content):
        """零样本解析"""
        if self.llm:
            log_content = log_content.replace('<*>', '<<preprocessed>>')

            messages = [
                {'role': 'system', 'content': system_message_zero_shot.format(f"{log_content}")}]
            template = self.process_one_log(log_content, messages)
        else:
            template = log_content
        return template

    def check_retry(self, result, log_content, messages):
        """
            检查对话结果是否需要重试
        """
        pattern = r"`+([^`]+)`+"
        matches = re.findall(pattern, result)
        messages.append({'role': 'assistant', 'content': result})
        if len(matches) != 1:
            messages.append({'role': 'user', 'content': retry_noresult_message})
            return result, messages
        result = matches[0]
        template_adjust = adjust_template(result)
        if not check_match(template_adjust, log_content):
            messages.append({'role': 'user', 'content': retry_nomatch_message})
            return result, messages
        return result, None
