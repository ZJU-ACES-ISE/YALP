import os

from evaluate.evaluator import  evaluate_ED, evaluate_UP, out_error_GA, out_error_PA, evaluate_FGA, calculate_PA, calculate_FTA
import pandas as pd
from postProcess import post_process

in_dir = '../datasets/2k_datasets'
in_dir = '../datasets/full_datasets'

datasets = ['BGL', 'HDFS', 'Linux', 'HealthApp', 'OpenStack', 'OpenSSH', 'Proxifier', 'HPC', 'Zookeeper', 'Mac',
            'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark']

datasets = []  # 重定义 数据集顺序

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 1000)


def output_csv(result, file):
    df_result = pd.DataFrame(result, columns=['dataset', 'precision', 'recall', 'f1', 'GA', 'MLA', 'ED', 'PTA', 'RTA', 'FTA', 'FGA', 'UP', 'tool_templates', 'ground_templates'])

    print(df_result)
    df_result.to_csv(file)
    return df_result


def remove_files(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)  # 删除文件


def evaluate_out(groundtruth_path, parsedresult_path, dataset, method, out_ga=True, out_pa=True, log_file=None, ):
    if out_ga:
        out_ga_path = f"{parsedresult_path}_error_ga.csv"
        remove_files([out_ga_path])  # 删除之前的GA结果
        out_error_GA(groundtruth_path, parsedresult_path, out_ga_path, dataset, log_file=log_file)  # 错误GA
    if out_pa:
        out_pa_path = f"{parsedresult_path}_error_pa.csv"
        remove_files([out_pa_path])  # 删除之前的MLA结果
        out_error_PA(groundtruth_path, parsedresult_path, out_pa_path, dataset, log_file=log_file)  # 错误GA

    # LogPub中计算FGA
    groundtruth = pd.read_csv(groundtruth_path, dtype=str)
    parsedresult = pd.read_csv(parsedresult_path, dtype=str)
    parsedresult.fillna("", inplace=True)

    tool_templates, ground_templates, FTA, PTA, RTA = calculate_FTA(dataset, groundtruth, parsedresult)
    PA = calculate_PA(groundtruth, parsedresult)
    ED = evaluate_ED(groundtruth, parsedresult)
    UP = evaluate_UP(groundtruth, parsedresult)
    GA, FGA = evaluate_FGA(groundtruth, parsedresult)

    # 计算指标，没有f1
    columns = ['method', 'dataset', 'GA', 'PA', 'ED', 'PTA', 'RTA', 'FTA', 'FGA', 'UP', 'tool_templates', 'ground_templates']
    result = [[method, dataset, GA, PA, ED, PTA, RTA, FTA, FGA, UP, tool_templates, ground_templates]]

    df_result = pd.DataFrame(result, columns=columns)
    return df_result


def evaluate_result(in_dir, out_dir, log_file, out_structured_path, method, out_ga=True, out_pa=False, run_log_file=None, dataset=""):
    """评估"""
    # 不进行后处理
    groundtruth_path = f"{in_dir}/{log_file}_structured.csv"
    parsedresult_path = out_structured_path

    # # 后处理调整
    adjust_out_structured_path = f"{out_dir}/{log_file}_structured_adjusted.csv"
    adjust_out_templates_path = f"{out_dir}/{log_file}_templates_adjusted.csv"
    post_process(out_structured_path, adjust_out_structured_path, adjust_out_templates_path)
    parsedresult_path = adjust_out_structured_path
    groundtruth_path = f"{in_dir}/{log_file}_structured_corrected_post.csv"

    # 开始评估
    print('=== Evaluation on %s ===' % dataset, file=run_log_file)
    df_result = evaluate_out(groundtruth_path, parsedresult_path, dataset, method=method, out_ga=out_ga, out_pa=out_pa, log_file=run_log_file)
    print(df_result.to_string(index=False), file=run_log_file)
    return df_result
