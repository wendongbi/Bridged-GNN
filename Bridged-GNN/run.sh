# Here are some examples to get the results shown in the paper: "Bridged-GNN: Knowledge Bridge Learning for Effective Knowledge Transfer"

# 1. twitter dataset, non-graph setting
# step-1: train bridged-graph learner
python main_bridged_graph.py --k_within 6 --k_cross 20 --num_epoch 400 --start_eval_epoch 300 --epsilon 0.5  --seed 0 --save --dataset_name twitter_unrelational --check_within 
# step-2: knowledge transfer on the learned bridged-graph with KTGNN
python main_graph_knowledge_transfer.py --num_layer 2 --hidden_dim 128 --path_data ../data_bridged_graph/twitter_unrelational_bridged_graph.dat --to_undirected 

# 2. office(amazon2dslr) dataset, non-graph setting
# step-1: train bridged-graph learner
python main_bridged_graph.py --hidden_dim 128 --num_epoch 400 --start_eval_epoch 300 --epsilon 0.5 --k_within 3 --k_cross 20  --seed 0  --save --dataset_name office_amazon2dslr --version v2  --check_within --check_cross 
# step-2: knowledge transfer on the learned bridged-graph with KTGNN
python main_graph_knowledge_transfer.py --num_layer 2 --hidden_dim 64 --path_data ../data_bridged_graph/office_amazon2dslr_bridged_graph.dat --to_undirected 

# 3. office(amazon2webcam) dataset, non-graph setting
# step-1: train bridged-graph learner
python main_bridged_graph.py --hidden_dim 128 --num_epoch 400 --start_eval_epoch 300 --epsilon 0.5 --k_within 3 --k_cross 8  --seed 0  --save --dataset_name office_amazon2webcam --version v2  --check_within --check_cross
# step-2: knowledge transfer on the learned bridged-graph with KTGNN
python main_graph_knowledge_transfer.py --num_layer 2 --hidden_dim 128 --path_data ../data_bridged_graph/office_amazon2webcam_bridged_graph.dat --to_undirected 

# 4. facebook(hamilton2caltech) dataset, cross-graph setting
# step-1: train bridged-graph learner
python main_bridged_graph.py --hidden_dim 64 --k_within 0 --k_cross 50 --num_epoch 400 --start_eval_epoch 300 --epsilon 0.5  --seed 0 --dataset_name fb_hamilton2caltech --check_within --check_cross  --thres_feat_sim 0.0
# step-2: knowledge transfer on the learned bridged-graph with KTGNN
python main_graph_knowledge_transfer.py --num_epoch 300 --num_layer 2 --hidden_dim 64 --path_data ../data_bridged_graph/fb_hamilton2caltech_bridged_graph.dat --to_undirected --no_dtc 

# 5. facebook(howard2simmons) dataset, cross-graph setting
# step-1: train bridged-graph learner
python main_bridged_graph.py --hidden_dim 64 --k_within 0 --k_cross 50 --num_epoch 400 --start_eval_epoch 300 --epsilon 0.5  --seed 0 --dataset_name fb_howard2simmons --check_within --check_cross  --thres_feat_sim 0.0 --eval_per_epoch 5
# step-2: knowledge transfer on the learned bridged-graph with KTGNN
python main_graph_knowledge_transfer.py --num_epoch 200 --num_layer 2 --hidden_dim 64 --path_data ../data_bridged_graph/fb_howard2simmons_bridged_graph.dat
