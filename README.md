# Auto-EDA Benchamark (A-EDA): Benchmark Dataset for Auto-generating EDA Sessions.
This repository contains the automatic benchmark used to evalute ATENA, a system for auto-generating Exploratory Data Analysis (EDA) sessions <a href="https://dl.acm.org/doi/abs/10.1145/3318464.3389779">  [Milo et al., SIGMOD '20] </a>. The repository is free for use for academic purposes. Upon using, please cite the paper:</br>
```Ori Bar El, Tova Milo, and Amit Somech. 2020. Automatically Generating Data Exploration Sessions Using Deep Reinforcement Learning. SIGMOD ’20, 1527–1537. DOI:https://doi.org/10.1145/3318464.3389779```

## The problem: Auto-generating meaningful, coherent EDA sessions.
Exploratory Data Analysis (EDA) is an essential yet highly demanding task. To get a head start before exploring a new dataset, data scientists often prefer to view existing EDA notebooks - illustrative, curated exploratory sessions, on the same dataset, that were created by fellow data scientists who shared them online. Unfortunately, such notebooks are not always available (e.g., if the dataset is new or confidential). 
</br>
As a first attempt (to our knowledge) to tackle this issue, we developed ATENA - a deep reinforcement learning (DRL) based system that takes an input dataset and auto-generates a compelling exploratory session, presented in an EDA notebook (See  <a href="https://dl.acm.org/doi/abs/10.1145/3318464.3389779"> the paper </a> for more details). 
