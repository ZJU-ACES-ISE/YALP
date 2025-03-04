import collections
import csv
import re
from collections import defaultdict

import pandas as pd
import scipy.special
from nltk.metrics.distance import edit_distance
from sklearn.metrics import accuracy_score
import numpy as np


def wla(array1, array2):
    count = 0
    word_pattern = re.compile(r'\w+')
    # 遍历数组的元素，逐个比较它们的单词部分
    for str1, str2 in zip(array1, array2):
        words1 = re.findall(word_pattern, str1)
        words2 = re.findall(word_pattern, str2)

        # 如果单词列表相同，则增加计数器的值
        if words1 == words2:
            count += 1
    return count / len(array1)


def evaluate(groundtruth, parsedresult):
    # 读取groundtruth和parsed log文件到dataframe中
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, index_col=False)


    null_logids = df_parsedlog[~df_parsedlog['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]

    # 计算精确字符串匹配的准确率
    accuracy_exact_string_matching = accuracy_score(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                                                    np.array(df_parsedlog.EventTemplate.values, dtype='str'))

    # 计算编辑距离
    edit_distance_result = []
    for i, j in zip(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                    np.array(df_parsedlog.EventTemplate.values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    # 计算编辑距离的均值和标准差
    edit_distance_result_mean = np.mean(edit_distance_result)

    pd.set_option('max_colwidth', 100)

    # 计算精确率、召回率、F1值和准确率
    (precision, recall, f_measure, accuracy_PA) = get_accuracy(df_groundtruth['EventTemplate'],
                                                               df_parsedlog['EventTemplate'])

    # 计算未出现事件的准确率
    unseen_events = df_groundtruth.EventTemplate.value_counts()
    df_unseen_groundtruth = df_groundtruth[df_groundtruth.EventTemplate.isin(
        unseen_events.index[unseen_events.eq(1)])]
    df_unseen_parsedlog = df_parsedlog[df_parsedlog.LineId.isin(
        df_unseen_groundtruth.LineId.tolist())]
    n_unseen_logs = len(df_unseen_groundtruth)
    if n_unseen_logs == 0:
        unseen_PA = 0
    else:
        unseen_PA = accuracy_score(np.array(df_unseen_groundtruth.EventTemplate.values, dtype='str'),
                                   np.array(df_unseen_parsedlog.EventTemplate.values, dtype='str'))
    # 打印评估结果

    mwla = wla(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
               np.array(df_parsedlog.EventTemplate.values, dtype='str'))
    print('WLA', mwla)

    print(
        'Precision: %.4f, Recall: %.4f, F1: %.4f, GA: %.4f, PA: %.4f, ED: %.4f, UP: %.4f' % (
            precision, recall, f_measure, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, unseen_PA))
    return precision, recall, f_measure, accuracy_PA, accuracy_exact_string_matching, edit_distance_result_mean, unseen_PA


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):

    # 计算groundtruth中每个事件的出现次数
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    # 计算groundtruth中有超过1个事件的真实配对数量
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    # 计算parsedlog中有超过1个事件的解析配对数量
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    # 初始化精确配对数量和正确解析的事件数量
    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    # 遍历parsedlog中的每个事件
    for parsed_eventId in series_parsedlog_valuecounts.index:
        # 找到包含该事件的日志ID
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        # 在groundtruth中找到与这些日志ID对应的事件
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts(        )
        # 如果groundtruth中只有1个事件,且日志数量相同,则正确解析
        error_eventIds = (            parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True

        # 解析对
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False

        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =',
                  error_eventIds, 'failed', logIds.size, 'messages')

        # 解析结果中的 解析答案 的对数
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)


    # 计算精确率、召回率和F1值
    precision = float(accurate_pairs) / \
                parsed_pairs if parsed_pairs != 0 else 0
    recall = float(accurate_pairs) / real_pairs if real_pairs != 0 else 0
    f_measure = 2 * precision * recall / \
                (precision + recall) if (precision + recall) != 0 else 0
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy


def out_error_GA(groundtruth, parsedresult, file_error_ga, dataset, log_file=None):
    print("\n\n\n____error:GA", file=log_file)
    f_error_ga = open(file_error_ga, 'a', newline='')
    csv_writer = csv.writer(f_error_ga)
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, index_col=False)

    null_logids = df_parsedlog[~df_parsedlog['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]

    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedlog['EventTemplate']
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()

    # 试 多模板解析为一组
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts(
        )
        # 解析对
        if series_groundtruth_logId_valuecounts.size != 1:
            print('\n\nL:' + parsed_eventId + ' ：  多模板解析为一组，需拆分', file=log_file)
            for template, count in series_groundtruth_logId_valuecounts.items():
                print(f"{template}\t{count}", file=log_file)
                csv_writer.writerow(
                    [dataset, '拆分', parsed_eventId, template, count])

    # 调试 解析多组
    for truth_eventId in series_groundtruth_valuecounts.index:
        logIds = series_groundtruth[series_groundtruth == truth_eventId].index
        series_parsedlog_logId_valuecounts = series_parsedlog[logIds].value_counts(
        )
        if series_parsedlog_logId_valuecounts.size != 1:
            print('\n\nT:' + truth_eventId + ' ：  一模板解析成多组，需合并', file=log_file)
            for template, count in series_parsedlog_logId_valuecounts.items():
                print(f"{template}\t{count}", file=log_file)
                csv_writer.writerow(
                    [dataset, '合并', truth_eventId, template, count])


def generate_mapping(queue1, queue2):
    mapping = defaultdict(list)
    for item1, item2 in zip(queue1, queue2):
        mapping[item1].append(item2)

    res = dict()
    for k, v in mapping.items():
        v_counted = collections.Counter(v)
        if len(v_counted) > 1 or v_counted[k] <= 0:
            res[k] = v_counted

    return res


def out_error_PA(groundtruth, parsedresult, file_error_PA, dataset, log_file=None):
    print("\n\n\n____error:PA", file=log_file)
    f_error_mla = open(file_error_PA, 'a', newline='')
    csv_writer = csv.writer(f_error_mla)

    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, index_col=False)

    null_logids = df_parsedlog[~df_parsedlog['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]

    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedlog['EventTemplate']

    mapping_result = generate_mapping(series_groundtruth, series_parsedlog)
    for k, v in mapping_result.items():
        print(f"\nT:{k}", file=log_file)
        for i, c in v.items():
            print(f"L:{i}\t{c}", file=log_file)
            csv_writer.writerow([dataset, k, i, c])


def evaluate_ED(df_groundtruth, df_parsedlog, filter_templates=None):
    null_logids = df_parsedlog[~df_parsedlog['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    # 计算编辑距离
    edit_distance_result = []
    for i, j in zip(np.array(df_groundtruth.EventTemplate.values, dtype='str'),
                    np.array(df_parsedlog.EventTemplate.values, dtype='str')):
        edit_distance_result.append(edit_distance(i, j))

    # 计算编辑距离的均值和标准差
    edit_distance_result_mean = np.mean(edit_distance_result)
    print('edit_distance (ED): %.4f' % (edit_distance_result_mean))
    return edit_distance_result_mean


def evaluate_UP(df_groundtruth, df_parsedlog, filter_templates=None):
    null_logids = df_parsedlog[~df_parsedlog['EventTemplate'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    # 计算未出现事件的准确率
    unseen_events = df_groundtruth.EventTemplate.value_counts()
    df_unseen_groundtruth = df_groundtruth[df_groundtruth.EventTemplate.isin(
        unseen_events.index[unseen_events.eq(1)])]
    df_unseen_parsedlog = df_parsedlog[df_parsedlog.LineId.isin(
        df_unseen_groundtruth.LineId.tolist())]
    n_unseen_logs = len(df_unseen_groundtruth)
    if n_unseen_logs == 0:
        unseen_PA = 0
    else:
        unseen_PA = accuracy_score(np.array(df_unseen_groundtruth.EventTemplate.values, dtype='str'),
                                   np.array(df_unseen_parsedlog.EventTemplate.values, dtype='str'))
    print('unseen_PA (UP): %.4f' % (unseen_PA))
    return unseen_PA


def evaluate_FGA(df_groundtruth, df_parsedlog, filter_templates=None):
    null_logids = df_parsedlog[~df_parsedlog['EventTemplate'].isnull()].index

    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    (GA, FGA) = get_accuracy_FGA(
        df_groundtruth['EventTemplate'], df_parsedlog['EventTemplate'])
    print('Grouping_Accuracy (GA): %.4f, FGA: %.4f,' % (GA, FGA))
    return GA, FGA


def get_accuracy_FGA(series_groundtruth, series_parsedlog, filter_templates=None):
    """ Compute accuracy metrics between log parsing results and ground truth

        Arguments
        ---------
            series_groundtruth : pandas.Series
                A sequence of groundtruth event Ids
            series_parsedlog : pandas.Series
                A sequence of parsed event Ids
            debug : bool, default False
                print error log messages when set to True

        Returns
        -------
            precision : float
            recall : float
            f_measure : float
            accuracy : float
        """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=[
        'groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('groundtruth')
    accurate_events = 0  # determine how many lines are correctly parsed
    accurate_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    for ground_truthId, group in grouped_df:
        series_parsedlog_logId_valuecounts = group['parsedlog'].value_counts()
        if filter_templates is not None and ground_truthId in filter_templates:
            for parsed_eventId in series_parsedlog_logId_valuecounts.index:
                filter_identify_templates.add(parsed_eventId)
        if series_parsedlog_logId_valuecounts.size == 1:
            parsed_eventId = series_parsedlog_logId_valuecounts.index[0]
            if len(group) == series_parsedlog[series_parsedlog == parsed_eventId].size:
                if (filter_templates is None) or (ground_truthId in filter_templates):
                    accurate_events += len(group)
                    accurate_templates += 1
    if filter_templates is not None:
        GA = float(accurate_events) / \
             len(series_groundtruth[series_groundtruth.isin(filter_templates)])
        PGA = float(accurate_templates) / len(filter_identify_templates)
        RGA = float(accurate_templates) / len(filter_templates)
    else:
        GA = float(accurate_events) / len(series_groundtruth)
        PGA = float(accurate_templates) / len(series_parsedlog_valuecounts)
        RGA = float(accurate_templates) / len(series_groundtruth_valuecounts)
    FGA = 0.0
    if PGA != 0 or RGA != 0:
        FGA = 2 * (PGA * RGA) / (PGA + RGA)
    return GA, FGA


def calculate_PA(groundtruth_df, parsedresult_df, filter_templates=None):
    if filter_templates is not None:
        groundtruth_df = groundtruth_df[groundtruth_df['EventTemplate'].isin(
            filter_templates)]
        parsedresult_df = parsedresult_df.loc[groundtruth_df.index]
    correctly_parsed_messages = parsedresult_df[['EventTemplate']].eq(
        groundtruth_df[['EventTemplate']]).values.sum()
    total_messages = len(parsedresult_df[['Content']])
    PA = float(correctly_parsed_messages) / total_messages
    print('Parsing_Accuracy (PA): {:.4f}'.format(PA))
    return PA


def calculate_FTA(dataset, df_groundtruth, df_parsedresult, filter_templates=None):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    correct_parsing_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()

    null_logids = df_parsedresult[~df_parsedresult['EventTemplate'].isnull(
    )].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedresult = df_parsedresult.loc[null_logids]

    series_groundtruth = df_groundtruth['EventTemplate']
    series_parsedlog = df_parsedresult['EventTemplate']
    series_groundtruth_valuecounts = series_groundtruth.value_counts()

    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=[
        'groundtruth', 'parsedlog'])
    grouped_df = df_combined.groupby('parsedlog')

    t1 = len(grouped_df) if filter_templates is None else len(filter_identify_templates)
    t2 = len(series_groundtruth_valuecounts) if filter_templates is None else len(filter_templates)
    print("Identify : {}, Groundtruth : {}".format(t1, t2))

    for identified_template, group in grouped_df:
        corr_oracle_templates = set(list(group['groundtruth']))

        if filter_templates is not None and len(corr_oracle_templates.intersection(set(filter_templates))) > 0:
            filter_identify_templates.add(identified_template)

        if corr_oracle_templates == {identified_template}:
            if (filter_templates is None) or (identified_template in filter_templates):
                correct_parsing_templates += 1

    if filter_templates is not None:
        PTA = correct_parsing_templates / len(filter_identify_templates)
        RTA = correct_parsing_templates / len(filter_templates)
    else:
        PTA = correct_parsing_templates / len(grouped_df)
        RTA = correct_parsing_templates / len(series_groundtruth_valuecounts)
    FTA = 0.0
    if PTA != 0 or RTA != 0:
        FTA = 2 * (PTA * RTA) / (PTA + RTA)

    print('PTA: {:.4f}, RTA: {:.4f} FTA: {:.4f}'.format(PTA, RTA, FTA))

    return t1, t2, FTA, PTA, RTA
