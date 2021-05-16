# REACT-IDA: benchmark dataset for real user analysis sesisons 
This Github repository contains (1) a collection of analysis sessions made by real users in the cyber security domain.
(2) a distance metric for analysis actions, results "displays" and analysis sessions, as described in the paper "Next-Step Suggestions for Modern Interactive Data Analysis Platforms".

The repository is free for use for academic purposes.
Upon using, please cite the following paper:

Tova Milo and Amit Somech. 2018. Next-Step Suggestions for Modern Interactive Data Analysis Platforms. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). ACM, New York, NY, USA, 576-585. DOI: https://doi.org/10.1145/3219819.3219848

## The Problem: Modern IDA Recommendations
Modern Interactive Data Analysis (IDA) platforms, such as Kibana,
Splunk, and Tableau, are gradually replacing traditional OLAP/SQL
tools, as they allow for easy-to-use data exploration,
visualization, and mining, even for users lacking SQL and
programming skills. Nevertheless, data analysis is still a difficult
task, especially for non-expert users.

There are two major challenges stems from the modern, often web-based analysis platforms:
1. IDA platforms facilitate composite analysis processes,
interweaving actions of  multiple types (filtering,
aggregation, pattern mining, visualization, etc.) while providing a
simplified syntax. 

2. In common IDA business environments, users (even of the same
department) often examine different datasets, for different
purposes. 


## Benchmark Dataset: Real-world IDA logs.
To our knowledge, there are no publicly available repositories of analysis actions performed on modern IDA platforms.
This benchmark dataset contains real-world analysis log in the domain of cyber security.

### Data Collection
We recruited 56 analysts, specializing in the domain of cyber-security (via dedicated forums, network security firms, and volunteer senior students from the Israeli National Cyber-Security Program), and asked them to analyze 4 different datasets using a prototype web-based analysis platform that we developed.
Each dataset, provided by the "Honeynet Project", contains between 350 to 13K rows of raw network logs that may reveal a distinct security event, e.g. malware communication hidden in network traffic, hacking activity inside a local network, an IP range/port scan, etc. (there is no connection between the tuples of different datasets).
The analysts were asked to perform as many analysis actions as required to reveal the details of the underlying security event of each dataset.

### How to use
The four datasets analyzed by the users are in ./raw_datasets.
The folder ./session_repositories contains the files: (1) actions.tsv, that contains the actions performed by the analysts, and (2) displays.tsv that contains a structural summary of the actions' result sets. 
Last, The folder ./lib contains helper scripts that allow to recreate the actions performed by the users and to examine their actual results as seen by the user at each point of the session.
Also, it contains the code of the distance metric for analysis actions and displays, as appear in the paper "Next-Step Suggestions for Modern Interactive Data Analysis Platforms"

We provide a Jupyter Notebook with a getting started code. 

 

