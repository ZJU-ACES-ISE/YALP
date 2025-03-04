# YALP


YALP is a zero-sample log parsing solution that combines LLM with traditional methods.

### Running
Install the required enviornment:
```
pip install -r requirements.txt
```

Add your LLM config in .env:
```
LLM_BASE_URL="https://xxx"
LLM_API_KEY="sk-xxx"
```

Run the following scripts to start the demo:

```
python main.py
```

### Benchmark

Running the benchmark script on Loghub_2k datasets, you could obtain the following results of GA.

| Dataset     | gpt-4o-mini-2024-07-18 | gpt-3.5-turbo-0125 | gpt-4-turbo-2024-04-09 |
|-------------|------------------------|--------------------|------------------------|
| HDFS        | 1                      | 1                  | 1                      |
| Hadoop      | 0.965                  | 0.974              | 0.974                  |
| Spark       | 0.9945                 | 0.9945             | 0.9965                 |
| Zookeeper   | 0.969                  | 0.969              | 0.969                  |
| BGL         | 0.97                   | 0.956              | 0.9765                 |
| HPC         | 0.9405                 | 0.9405             | 0.9405                 |
| Thunderbird | 0.9745                 | 0.9805             | 0.994                  |
| Windows     | 0.9965                 | 0.9895             | 0.995                  |
| Linux       | 0.9285                 | 0.923              | 0.9245                 |
| Android     | 0.9545                 | 0.9565             | 0.9555                 |
| HealthApp   | 1                      | 0.9775             | 1                      |
| Apache      | 1                      | 1                  | 1                      |
| Proxifier   | 0.5265                 | 0.5265             | 0.5265                 |
| OpenSSH     | 0.741                  | 0.741              | 0.741                  |
| OpenStack   | 0.9925                 | 0.9925             | 0.9925                 |
| Mac         | 0.7745                 | 0.826              | 0.8385                 |

### Citation

If you use the code or benchmarking results in your publication, please kindly cite the following papers.

+ [**ICWS'24**] C Zhi, L Cheng, M Liu, X Zhao, Y Xu, S Deng. [LLM-powered Zero-shot Online Log Parsing](https://ieeexplore.ieee.org/abstract/document/10707403), *IEEE International Conference on Web Services(ICWS)*, 2024.
