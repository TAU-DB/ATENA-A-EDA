
## The A-EDA Benchmark.
The benchmark includes 8 datasets from two different schemas: flight-delays and cyber-security.
Each dataset is accompanied with a "gold-standard" set of EDA sessions, made by expert data scientists in order to showcase the dataset highlights in a coherent, easy-to-follow mannar.
The gold-standard EDA sessions comprise FILTER and GROUP-BY operations only.

### Evaluation metrics
The A-EDA benchmark draws similar lines to machine-translation and caption-generation benchmarks.
We have 5 different supported metrics:

#### 1. Precision: 
This measure compares consider the EDA notebooks as sets of distinct views (ignoring their order), counting a "hit" if a view occurs in one of the gold-standard notebooks and a "miss" otherwise. It is calculated by `hits/(hits+misses)`

#### 2-4. T-BLEU-1, T-BLEU-2, T-BLEU-3:
These measures are based on the well known BLEU score, used for comparing sentences in image captions and machine translations (In our case the "sentence" is the sequence of views in the notebook). T-BLEU is more strict than Precision, since it also considers the prevalence of each view in the gold-standard set, as well as their order, by comparing subsequences of size `n` (rather than single views). We use `n` between 1 to 3 for T-BLEU-1 through T-BLEU-3.

#### 5. EDA-SIM: 
We use a dedicated distance metric devised in <a href= "https://www.kdd.org/kdd2018/accepted-papers/view/next-step-suggestions-for-modern-interactive-data-analysis-platforms"> [Milo,Somech, KDD '18] </a>  to estimate the similarity for exploratory sessions (the source code is available on <a href="https://github.com/TAU-DB/REACT-IDA-Recommendation-benchmark"> GitHub repository </a>). EDA-SIM also considers the order of views yet allows for a fine-grained comparison of their content (i.e., almost identical views are considered "misses'' in the above measures, yet EDA-SIM will evaluate them as highly similar). For the final EDA-Sim score, we compare the generated notebook to each of the Gold-Standard notebooks and take the maximal EDA-SIM score obtained. 




## Getting Started.
A very intuitive getting started notebook is available <a href="https://github.com/TAU-DB/ATENA-A-EDA/blob/master/notebook_example.ipynb"> here </a>. 
In this notebook we show how the datasets can be accessed, the syntax for the EDA opertions, and how to use the evaluation metrics. 


