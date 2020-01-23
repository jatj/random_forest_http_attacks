# Random forest for HTTP DDoS attacks

This repo contains the implementation of a Random Forest algorithm for classifying network flows into normal or attack flows. To run a pretrained model with the first 5000 flows of the dataset run `python3 main.py -m load_random_forest_bin`. If you want to train the whole model again run `python3 main.py -m random_forest_bin`.