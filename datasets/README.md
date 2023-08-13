# Datsets
We preprocessed all datasets used in this paper, including relational and unrelational datasets,  into the format of *torch_geometric*. And our processed datasets are available at `./dataset/xxx.dat` (unzip the file named 'dataset_pyg.tar.gz' in the dataset folder first). And we provide a demo to load the dataset.

​		The dataset used in this paper includes unrealtional data (Twitter, Office31(A2W), office31(A2D)), Sync-$UD$) and relational data (FB(Hamilton2Caltech), FB(Howard2Simmons), Sync-$RD_{intra}$, $\text{Twitter}_\text{Graph}$, Company, Sync-$RD_{intra+inter}$). And these datasets are used to validate the three scenarios ($UD$, $RD_{intra}$, $RD_{intra+inter}$) proposed in the paper.

​		Besides, we provide a demo (data_viewer.ipynb) to load these datasets with pytorch. And the full code will be released at the publication time of this paper.

