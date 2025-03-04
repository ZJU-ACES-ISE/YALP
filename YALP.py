# -*- coding: utf-8 -*-
import math
from collections import Counter
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from itertools import islice

from DataHandler import DataHandler
from Sample import Sample
from Template import Template
from LLM import LLM
from LLMParser import LLMParser
from LogFormatParser import LogFormatParser
from merge_template import get_similar_templates, merge_templates_by_lcs
from split_template import split_template_by_lcs
from utils.utils_file import check_dir
from utils.utils_match import special_character_replace, replace_content, unique_wildcard, check_one_word, check_word, check_match, special_character_restore


class YALP:

    def __init__(self, dataset_regexes: List[str], pre_regexes: List[str],
                 llm_model: Optional[LLM] = None,
                 split_enable: bool = True, split_limit: int = 2,
                 merge_enable: bool = True, prune_limit=0.3, cluster_enable: bool = True,
                 include_enable: bool = False,
                 line_limit: Optional[int] = None):

        self.dataset_regexes: List[str] = dataset_regexes  # 数据集对应的规则
        self.pre_regexes: List[str] = pre_regexes  # 预处理的规则
        self.llm_parser: LLMParser = LLMParser(llm_model)  # 大模型解析器
        self.split_enable = split_enable  # 是否进行模板拆分
        self.split_limit = split_limit  # 拆分阈值
        self.merge_enable = merge_enable  # 是否进行模板合并
        self.prune_limit = prune_limit if merge_enable else 0
        self.cluster_enable = cluster_enable if merge_enable else False  # 是否进行聚类
        self.include_enable = include_enable  # 是否允许模板存在包含关系
        self.line_limit = line_limit

        self.dh: DataHandler = DataHandler()  # 数据处理
        self.log_format_parser: LogFormatParser = LogFormatParser()

        # 列表参数
        self.logs: List[Dict] = []  # 日志头
        self.sample_ids: List[str] = []  # 日志对应的样本
        self.log_file = None

    def close(self) -> None:
        if self.llm_parser.llm:
            self.llm_parser.llm.save()

    def parse_file(self, in_log_path: str, out_structured_path: str, out_templates_path: str, log_format: str, log_file) -> None:
        """
            日志解析，批量解析，以文件为输入，以文件为输出
        """
        self.log_file = log_file
        self.log_format_parser.generate_logformat_regex(log_format)  # 解析日志头的解析器

        with open(in_log_path, 'r', encoding='utf-8') as in_file:  # 读取日志文件
            in_file = islice(in_file, self.line_limit) if self.line_limit else in_file  # 数量限制
            for i, log_line in tqdm(enumerate(in_file), desc="parse processing", unit=" Lines", total=self.line_limit, mininterval=1, file=log_file):
                log = self.log_format_parser.log_to_dict(log_line.strip())  # 解析日志头
                self.logs.append(log)
                if 'Content' in log.keys():  # 如果未得到有效的日志内容，则跳过
                    self.parse_content(log['Content'])
        self.output_result(out_structured_path, out_templates_path)

    def parse_content(self, log_content: str) -> None:
        """
            单条日志解析，输入仅为日志内容（已去日志头）
        """
        log_content = self.preprocess(log_content)  # 预处理
        sample = self.add_sample(log_content)  # 添加样本，如果已有重复样本，则直接返回

        if sample.template_id is not None:  # 已有模板
            template = self.dh.select_template_by_id(sample.template_id)  # 获取模板
        else:
            template = self.search_template(log_content)  # 正则表达式匹配已有模板
            if template is None:  # 匹配不到已有模板，需解析生成新模板
                template = self.parse_new(log_content)
            sample.template_id = template.template_id

        if self.split_enable:
            self.template_split(template)  # 模板拆分

    def preprocess(self, log_content: str) -> str:
        """
            预处理：特殊字符处理、数据集对应的规则替换、预处理的规则替换、去除重复连续的通配符
        """
        log_content = special_character_replace(log_content)  # 特殊字符处理
        log_content = replace_content(log_content, self.dataset_regexes, wildcard='<*>')  # 数据集对应的规则替换
        log_content = replace_content(log_content, self.pre_regexes, wildcard='<*>')  # 预处理的规则替换
        log_content = unique_wildcard(log_content, wildcard='<*>')  # 去除重复连续的通配符
        return log_content

    def parse_new(self, log_content: str) -> Template:
        """
            解析日志，得到新模版
        """
        if check_one_word(log_content):  # 没有或者只有一个单词的日志，不进行解析
            template_without_id = Template(log_content, sample_limit=self.split_limit)
        else:
            log_template = self.llm_parser.parse_message_zero_shot(log_content)  # 大语言模型解析，得到初始日志模板
            template_without_id = Template(log_template, sample_limit=self.split_limit)

            if self.merge_enable:  # 模板合并
                template_without_id = self.template_merge(template_without_id)
        return self.add_template(template_without_id)  # 添加模板，并提供模板id

    def template_merge(self, template_without_id):
        """模板合并"""
        templates = self.dh.select_templates()  # 获取所有的日志模板
        similar_templates = get_similar_templates(templates, template_without_id, self.cluster_enable)  # 获取相似模板
        merged_template = merge_templates_by_lcs(similar_templates)  # 模板合并
        template_without_id = merged_template if check_word(merged_template.template_text) else template_without_id  # 模板合并结果为空，则不进行合并
        return template_without_id

    def template_split(self, template: Template) -> None:
        """
            模板拆分
        """
        len_sample = self.dh.select_sample_len_by_template_id(template.template_id)  # 获取模板的样本数量
        if len_sample >= template.sample_limit:  # 达到样本数量
            samples = self.dh.select_samples_by_template_id(template.template_id)  # 获取模板的样本
            split_results = split_template_by_lcs(samples, round(math.log(len_sample, self.split_limit)))  # 计算模板拆分结果
            self.transfer_sample(template, split_results)  # 转移样本

    def transfer_sample(self, template, split_results):
        if len(split_results) == 1:
            template.sample_limit = split_results[0][0].sample_limit  # 更新模板的样本数量
        else:
            self.dh.empty_template(template)  # 原模板设置空
            for new_template_without_id, samples in split_results:
                new_template = self.add_template(new_template_without_id)  # 插入新模板
                for sample in samples:
                    sample.template_id = new_template.template_id  # 更新样本的模板id
                    self.dh.update_template_id_of_sample(new_template, sample)  # 更新样本的模板id
            self.dh.remove_template(template)  # 删除原模板

    def search_template(self, log_content: str) -> Optional[Template]:
        """
            搜索可匹配日志的模板
        """
        templates = self.dh.select_templates()  # 获取所有的日志模板
        for template in templates:  # 对每一个日志模板匹配
            if check_match(template.template_text, log_content):
                return template
        for template in templates:  # 对每一个日志模板匹配
            if check_match(template.template_text.replace('{V}', ''), log_content, ):
                return template
        return None

    def add_template(self, template_without_id: Template) -> Template:
        """
            插入新模板，并检验是否存在包含关系
        """
        template_without_id.template_text = unique_wildcard(template_without_id.template_text)
        if not self.include_enable:  # 不允许模板存在包含关系，则需要检验
            template = self.check_include(template_without_id)  # 检验模板之间的包含覆盖关系,并返回新模板或返回原模板
        else:
            template = self.dh.insert_template(template_without_id)  # 插入新模板
        return template

    def check_include(self, new_template_without_id: Template) -> Template:
        """
            检验模板之间的包含覆盖关系,并返回新模板或返回原模板
        """
        templates = self.dh.select_templates()
        sub_templates = []
        new_template = None
        for template in templates:  # 遍历所有模板
            if template.template_text == new_template_without_id.template_text:  # 模板相同
                return template
            elif check_match(template.template_text, new_template_without_id.template_text):  # 新模板被覆盖
                new_template = template
            elif check_match(new_template_without_id.template_text, template.template_text):  # 新模板覆盖子模板
                sub_templates.append(template)

        if new_template is None:  # 模板不被覆盖,则新建
            new_template = self.dh.insert_template(new_template_without_id)  # 插入新模板
        for sub_template in sub_templates:  # 删除能覆盖的子模板，并将样本转移到父模板
            self.dh.update_template_id_of_sample_by_template(new_template, sub_template)  # 转移样本
            self.dh.remove_template(sub_template)  # 删除子模板
        return new_template

    def add_sample(self, log_content: str) -> Sample:
        """
            添加样本, 如果已有重复样本，则直接返回
        """
        sample = self.dh.select_sample_by_text(log_content)  # 查找是否存在相同的样本
        if sample is not None:  # 如果存在，则增加数量
            sample.log_num += 1
            sample = self.dh.update_log_num_of_sample(sample)  # 更新数量
        else:  # 如果不存在，插入新样本
            sample_without_id = Sample(log_content)
            sample = self.dh.insert_sample(sample_without_id)  # 插入新样本
        self.sample_ids.append(sample.sample_id)  # 记录样本id
        return sample

    def output_result(self, out_structured_path, out_templates_path):
        """
            输出结果
        """
        check_dir(out_structured_path)  # 检查输出目录是否存在
        check_dir(out_templates_path)  # 检查输出目录是否存在

        log_df = pd.DataFrame(columns=self.log_format_parser.headers, data=self.logs)  # 输出结构化日志
        log_df['LineId'] = range(1, len(log_df) + 1)
        samples = [self.dh.select_sample_by_id(sample_id) for sample_id in self.sample_ids]  # 获取现有样本
        event_ids = [sample.template_id for sample in samples]  # 获取现有样本对应的模板id
        log_df['EventId'] = event_ids
        event_templates = [self.dh.select_template_by_id(event_id).template_text for event_id in event_ids]  # 获取现有样本对应的模板
        event_templates = [special_character_restore(event_template) for event_template in event_templates]  # 恢复特殊字符
        log_df['EventTemplate'] = event_templates
        log_df.to_csv(out_structured_path)

        unique_templates = sorted(Counter(event_templates).items(), key=lambda k: k[1], reverse=True)
        temp_df = pd.DataFrame(unique_templates, columns=['EventTemplate', 'Occurrences'])
        temp_df.to_csv(out_templates_path)
