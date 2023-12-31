# Bridged-GNN
Bridged-GNN for Knowledge Bridge Learning (KBL), the paper is accepted by CIKM2023.



## Datasets

We conduct experiments on five datasets, including four real-world datasets and  a synthetic dataset. We preprocessed all datasets used in this paper, including relational and unrelational datasets,  into the format of *torch_geometric*. And our processed datasets are available at `./dataset/xxx.dat` (unzip the file named 'dataset_pyg.tar.gz' in the dataset folder first). And we provide a demo to load the dataset. The detailed information of these datasets is as follows:

* **Twitter**: Twitter is a social network dataset that describes the social relations among politicians (source domain) and civilians (target domain) on the Twitter platform. The node features are the  GloVe  embedding of the user's Tweets (we do not use the user profile features of politicians used in the original paper), and the goal is to predict the political tendency of civilians. The original dataset (Twitter_Graph) is used in RD_{intra&inter} scenarios. By removing all original edges, we also construct the Twitter_UD dataset in un-relational data ($UD$) scenarios. 

* **Office31**: The Office31 dataset is a mainstream image benchmark for transfer learning, which contains 31 categories with a total of 4,652 images. There are three different domains (Amazon, Webcam, and DSLR) in this dataset. We select Webcam and DSLR, which are data-hungry, as the target domain, and use Amazon, which is rich in data, as the source domain. Then we construct two datasets: **Office31 (A$\rightarrow$W)** and **Office31 (A$\rightarrow$D)**. For the two datasets, all source domain samples  are used for training, and each category in the target domain randomly selects 3 samples for training, and the remaining samples are further divided into the validation set and test set in equal proportions. 

* **FB**: FB (Facebook100) datasets are social networks of 100 different universities in the United States. Each sample (students or faculties) have six attributes (e.g., gender, major, and dorm), and the goal is to predict Node identity flags. We view the social network of each university as a  different domain and select two of the university’s social networks and put them together to form a cross-network dataset in $RD_{intra}$ scenarios. In this paper, we select four universities and construct two cross-network datasets: FB (Hamilton$\rightarrow$Caltech) and FB (Howard$\rightarrow$Simmons). 

* **Company**: The Company dataset is a company investment network with 10641 real-world companies in China. The node attributes are the company's financial attributes. We regard listed companies as source domain and unlisted companies as target domain and the goal is to predict the risk status of non-listed companies. We complement missing features of non-listed companies by the mean value of neighbors (we do not consider the feature missing problems in the original paper ). And we use this dataset in the RD_{intra&inter} scenario.

* **Sync-UD/Sync-RD_{intra}/Sync-RD_{intra&inter}**: We also construct a synthetic dataset for three scenarios of GKT by randomly sampling points of source and target domains  from two distinct Multivariate Gaussian distributions. Samples of the source and target domains in this dataset are designed to have distinct conditional distribution and marginal distribution to validate our motivations.  We use this dataset in all three scenarios, and we randomly add edges with a fixed homophilous ratio of 70% for **Sync-RD_{intra}** and **Sync-RD_{intra&inter}**.

