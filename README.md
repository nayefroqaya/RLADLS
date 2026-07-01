# LogDRL : Submitted in :The International Conference on Cooperative Information Systems (CoopIS) -2026

a cost-aware, semi-supervised, deep reinforcement learning (RL) framework for early anomaly detection in system logs.

## Project Structure

```text

LogDRL/
├── src/
│   ├── __init__.py
│   ├── cross_dataset.py        # Cross-dataset training/evaluation utilities
│   ├── data.py                 # Data loading and preprocessing
│   ├── env.py                  # Reinforcement-learning environment
│   ├── evaluate.py             # Evaluation metrics and test procedure
│   ├── experiment.py           # Experiment execution pipeline
│   ├── model.py                # Deep learning and DQN model definitions
│   ├── replay_buffer.py        # Experience replay memory for DQN training
│   ├── slm_embedder.py         # Semantic log representation using SLM embeddings
│   ├── train.py                # Training procedure
│   └── utils.py                # Helper functions
│
├── check_labels.py             # Utility script for checking dataset labels
├── config.yaml                 # Main configuration file
├── main.py                     # Main entry point for running experiments
├── README.md                   # Project documentation
├── requirements.txt            # Required Python packages
```


## 📊 Datasets
We used two open-source log datasets (more will be added in the future):

| Software System     | Description                        | Data Size| Link                                         |
|--------------------|------------------------------------|-----------|----------------------------------------|
| HDFS               | Hadoop Distributed File System log | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)   |
| BGL                | Blue Gene/L supercomputer log      | 708.76 MB | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (1G)   | Thunderbird supercomputer log      | 1 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Spirit (SP_150MB)  | Supercomputing system log          | 150 MB    | [Figshare](https://figshare.com/s/6d3c6a83f4828d17be79?file=27775929) |


---

## ⚙️ Environment
All libraries are specified with their versions in the requirements file (e.g., Main path/requirements.txt).

```bash
pip install -r requirements.txt
```

---
## 🛠️ Preparation - Parsing step:
Steps to run LogSLM:

1. Install all required libraries from the requirements file (e.g., Main path/requirements.txt).
2. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`,`TH_1G`, `SP_150MB`) and upload the (datasetname.log) to this directory.
3. In main.py, set the dataset name (e.g., `HDFS`, `BGL`,`TH_1G`, `SP_150MB`)
4. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
5. The parsing code is available in the `drain_parser` folder.
6. Specify the dataset name in `demo.py` (e.g., BGL). The code is available for all datasets. Uncomment the lines of the dataset you need to use
7. For data parsing, all libraries are specified with their versions in the requirements file (e.g., drain_parser/requirements.txt). 
8. To start the parsing process, run (drain_parser/demo.py). 
9. The parsing output will be generated and saved in the datasets' directory.
10. The output of Drain is CSV file. 

---
## 🛠️ Preparation - Data Splitting  step:

1. After the parsing, we run load_datalog.py
2. We run the  main.py: 

## Data Format
The input data should be provided as preprocessed `.pkl` or `.csv` split files.
The required columns are configured in the YAML file:

Example split files:

```text
../datasets/BGL/1_BGL_Splitted_Datasets/train_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/val_df.pkl
../datasets/BGL/1_BGL_Splitted_Datasets/test_df.pkl
```

## 📬 Contact
We are happy to answer your questions:   

| Name               | Email Address                             |
|--------------------|-------------------------------------------|
| Nayef Roqaya       | roqaya@staff.uni-marburg.de               |
| Thorsten Papenbrock| papenbrock@informatik.uni-marburg.de      |