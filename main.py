import os
import pandas as pd

from LLM import ChatGPTLLM
from YALP import YALP
from evaluate.evaluate import evaluate_result

"""预定义的正则表达式"""
pre_regexes = [
    r'(?:[\w-]+\.)+(?:com|cn|hk|uk|org|net|gov|nl|at|edu|io|asia)',
    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', r'([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}', r'(\d+\.){3}\d+',
    r'[+-]?\b\d+(\.\d+)?\b', r'\b0[xX][\da-fA-F]+\b',
    r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b',
    r'root'
]

"""基准数据集设置"""
benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": ["blk_-?\\d+", "(\\d+\\.){3}\\d+(:\\d+)?"]
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
        "log_format": "<Date> <Time> <Level> \\[<Process>\\] <Component>: <Content>",
        "regex": ["(\\d+\\.){3}\\d+"]
    },
    "Spark": {
        "log_file": "Spark/Spark_2k.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": ["(\\d+\\.){3}\\d+", "\\b[KGTM]?B\\b", "([\\w-]+\\.){2,}[\\w-]+"]
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
        "log_format": "<Date> <Time> - <Level>  \\[<Node>:<Component>@<Id>\\] - <Content>",
        "regex": ["(/|)(\\d+\\.){3}\\d+(:\\d+)?"]
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": ["core\\.\\d+"]
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": []
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\\[<PID>\\])?: <Content>",
        "regex": ["(\\d+\\.){3}\\d+"]
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "regex": ["0x.*?\\s"]
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\\[<PID>\\])?: <Content>",
        "regex": ["(\\d+\\.){3}\\d+", "\\d{2}:\\d{2}:\\d{2}"]
    },
    "Android": {
        "log_file": "Android/Android_2k.log",
        "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        "regex": ["(/[\\w-]+)+", "([\\w-]+\\.){2,}[\\w-]+", "([-+]?\\b\\d+\\b)|\\b0[Xx][a-fA-F\\d]+\\b|\\b[a-fA-F\\d]{4,}\\b"]
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
        "log_format": "<Time>\\|<Component>\\|<Pid>\\|<Content>",
        "regex": []
    },
    "Apache": {
        "log_file": "Apache/Apache_2k.log",
        "log_format": "\\[<Time>\\] \\[<Level>\\] <Content>",
        "regex": ["(\\d+\\.){3}\\d+"]
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_2k.log",
        "log_format": "\\[<Time>\\] <Program> - <Content>",
        "regex": ["<\\d+\\ssec", "([\\w-]+\\.)+[\\w-]+(:\\d+)?", "\\d{2}:\\d{2}(:\\d{2})*", "[KGTM]B"]
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\\[<Pid>\\]: <Content>",
        "regex": ["(\\d+\\.){3}\\d+", "([\\w-]+\\.){2,}[\\w-]+"]
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \\[<ADDR>\\] <Content>",
        "regex": ["((\\d+\\.){3}\\d+,?)+"]
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\\[<PID>\\]( \\(<Address>\\))?: <Content>",
        "regex": ["([\\w-]+\\.){2,}[\\w-]+"]
    }
}


if __name__ == "__main__":
    df_results = []
    datasetSource = "2k_datasets"
    method = "YALP"
    in_dir = f"datasets/{datasetSource}"
    out_dir = f"outputs/{datasetSource}/{method}"
    for dataset, setting in benchmark_settings.items():
        print('\n=== Evaluation on %s ===' % dataset)

        out_dataset_dir = f"{out_dir}/{dataset}"
        if not os.path.exists(out_dataset_dir):
            os.makedirs(out_dataset_dir)
        in_log_path = f"{in_dir}/{setting['log_file']}"
        out_structured_path = f"{out_dir}/{setting['log_file']}_structured.csv"
        out_templates_path = f"{out_dir}/{setting['log_file']}_templates.csv"
        run_log_file = open(f"{out_dataset_dir}/run.log", 'w')

        parser = YALP(dataset_regexes=setting['regex'], pre_regexes=pre_regexes,
                      # llm_model=None,
                      llm_model=ChatGPTLLM(model="gpt-4o-mini-2024-07-18", cache_path="gpt-4o-mini-2024-07-18.json"),
                      # llm_model=ChatGPTLLM(model="gpt-3.5-turbo-0125", cache_path="gpt-3.5-turbo-0125.json"),
                      # llm_model=ChatGPTLLM(model="gpt-4-turbo-2024-04-09", cache_path="gpt-4-turbo-2024-04-09.json"),
                      split_enable=True, split_limit=2,
                      merge_enable=True, prune_limit=0.3, cluster_enable=True,
                      include_enable=False,
                      line_limit=None)
        try:
            parser.parse_file(in_log_path, out_structured_path, out_templates_path, setting['log_format'], log_file=run_log_file)
        finally:
            if parser:
                parser.close()


        df_result = evaluate_result(in_dir, out_dir, setting['log_file'], out_structured_path, method, out_ga=True, out_pa=True, run_log_file=run_log_file, dataset=dataset)
        df_results.append(df_result)

    df = pd.concat(df_results, axis=0)
    print(df)
    out_result_path = f"{out_dir}/evaluate_adjusted.csv"
    df.to_csv(out_result_path)
