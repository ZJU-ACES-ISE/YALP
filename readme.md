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

Running the benchmark script on Loghub_2k datasets, you could obtain the following results of GA and PA.

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">gpt-3.5-turbo-0125</th>
      <th colspan="2">gpt-4o-mini-2024-07-18</th>
      <th colspan="2">gpt-4-turbo-2024-04-09</th>
    </tr>
    <tr>
      <th>GA</th>
      <th>PA</th>
      <th>GA</th>
      <th>PA</th>
      <th>GA</th>
      <th>PA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HDFS</td>
      <td>1</td>
      <td>0.9425</td>
      <td>1</td>
      <td>0.9425</td>
      <td>1</td>
      <td>0.9425</td>
    </tr>
    <tr>
      <td>Hadoop</td>
      <td>0.974</td>
      <td>0.305</td>
      <td>0.965</td>
      <td>0.4685</td>
      <td>0.974</td>
      <td>0.5615</td>
    </tr>
    <tr>
      <td>Spark</td>
      <td>0.9945</td>
      <td>0.936</td>
      <td>0.9945</td>
      <td>0.937</td>
      <td>0.9965</td>
      <td>0.954</td>
    </tr>
    <tr>
      <td>Zookeeper</td>
      <td>0.969</td>
      <td>0.6505</td>
      <td>0.969</td>
      <td>0.4985</td>
      <td>0.969</td>
      <td>0.833</td>
    </tr>
    <tr>
      <td>BGL</td>
      <td>0.956</td>
      <td>0.415</td>
      <td>0.97</td>
      <td>0.454</td>
      <td>0.9765</td>
      <td>0.452</td>
    </tr>
    <tr>
      <td>HPC</td>
      <td>0.9405</td>
      <td>0.5725</td>
      <td>0.9405</td>
      <td>0.8475</td>
      <td>0.9405</td>
      <td>0.876</td>
    </tr>
    <tr>
      <td>Thunderbird</td>
      <td>0.9805</td>
      <td>0.503</td>
      <td>0.9745</td>
      <td>0.9115</td>
      <td>0.994</td>
      <td>0.934</td>
    </tr>
    <tr>
      <td>Windows</td>
      <td>0.9895</td>
      <td>0.3125</td>
      <td>0.9965</td>
      <td>0.3155</td>
      <td>0.995</td>
      <td>0.3195</td>
    </tr>
    <tr>
      <td>Linux</td>
      <td>0.923</td>
      <td>0.3615</td>
      <td>0.9285</td>
      <td>0.362</td>
      <td>0.9245</td>
      <td>0.3895</td>
    </tr>
    <tr>
      <td>Android</td>
      <td>0.9565</td>
      <td>0.6705</td>
      <td>0.9545</td>
      <td>0.678</td>
      <td>0.9555</td>
      <td>0.6815</td>
    </tr>
    <tr>
      <td>HealthApp</td>
      <td>0.9775</td>
      <td>0.63</td>
      <td>1</td>
      <td>0.65</td>
      <td>1</td>
      <td>0.6565</td>
    </tr>
    <tr>
      <td>Apache</td>
      <td>1</td>
      <td>0.978</td>
      <td>1</td>
      <td>0.56</td>
      <td>1</td>
      <td>0.978</td>
    </tr>
    <tr>
      <td>Proxifier</td>
      <td>0.5265</td>
      <td>0.895</td>
      <td>0.5265</td>
      <td>0.895</td>
      <td>0.5265</td>
      <td>0.895</td>
    </tr>
    <tr>
      <td>OpenSSH</td>
      <td>0.741</td>
      <td>0.6705</td>
      <td>0.741</td>
      <td>0.667</td>
      <td>0.741</td>
      <td>0.737</td>
    </tr>
    <tr>
      <td>OpenStack</td>
      <td>0.9925</td>
      <td>0.451</td>
      <td>0.9925</td>
      <td>0.462</td>
      <td>0.9925</td>
      <td>0.4725</td>
    </tr>
    <tr>
      <td>Mac</td>
      <td>0.826</td>
      <td>0.4755</td>
      <td>0.7745</td>
      <td>0.4965</td>
      <td>0.8385</td>
      <td>0.5575</td>
    </tr>
    <tr>
      <td>Avg</td>
      <td>0.9217</td>
      <td>0.6106</td>
      <td>0.9205</td>
      <td>0.6341</td>
      <td>0.9265</td>
      <td>0.7025</td>
    </tr>
  </tbody>
</table>

### Citation

If you use the code or benchmarking results in your publication, please kindly cite the following papers.

+ [**ICWS'24**] C Zhi, L Cheng, M Liu, X Zhao, Y Xu, S Deng. [LLM-powered Zero-shot Online Log Parsing](https://ieeexplore.ieee.org/abstract/document/10707403), *IEEE International Conference on Web Services(ICWS)*, 2024.
