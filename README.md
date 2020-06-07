# ATENA Auto-EDA Benchamark (A-EDA): A benchmark dataset for auto-generating EDA sessions.
This repository contains the automatic benchmark used to evalute ATENA, a system for auto-generating Exploratory Data Analysis (EDA) sessions <a href="https://dl.acm.org/doi/abs/10.1145/3318464.3389779">  [Milo et al., SIGMOD '20] </a>
</br>
## The problem: Auto-generating meaningful, coherent EDA sessions.
Exploratory Data Analysis (EDA) is an essential yet highly demanding task. To get a head start before exploring a new dataset, data scientists often prefer to view existing EDA notebooks - illustrative, curated exploratory sessions, on the same dataset, that were created by fellow data scientists who shared them online. Unfortunately, such notebooks are not always available (e.g., if the dataset is new or confidential). 
</br>
As a first attempt (to our knowledge) to tackle this issue, we developed ATENA - a deep reinforcement learning (DRL) based system that takes an input dataset and auto-generates a compelling exploratory session, presented in an EDA notebook (See  <a href="https://dl.acm.org/doi/abs/10.1145/3318464.3389779"> the paper </a> for more details). 

## The A-EDA Benchmark.
The benchmark includes 8 datasets from two different schemas: flight-delays and cyber-security.
Each dataset is accompanied with a "gold-standard" set of EDA sessions, made by expert data scientists in order to showcase the dataset highlights in a coherent, easy-to-follow mannar.
The gold-standard EDA sessions comprise FILTER and GROUP-BY operations only.

### Evaluation metrics
The A-EDA benchmark draws similar lines to machine-translation and caption-generation benchmarks.
We have 5 different supported metrics:

####1. Precision:
This measure compares consider the EDA notebooks as sets of distinct views (ignoring their order), counting a "hit" if a view occurs in one of the gold-standard notebooks and a "miss" otherwise. It is calculated by $'\frac{\text{hits}}{\text{hits+misses}}'$




## Getting Started.
