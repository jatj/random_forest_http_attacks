# Random forest for HTTP DDoS attacks

This repo contains the implementation of a Random Forest algorithm for classifying network flows into normal or attack flows. To run a pretrained model with the first 5000 flows of the dataset run `python3 main.py -m load_random_forest_bin`. If you want to train the whole model again run `python3 main.py -m random_forest_bin`.

In this repo we used a HTTP DDoS attack dataset from , you can check more of this dataset [here](https://www.unb.ca/cic/datasets/dos-dataset.html) and download it [here](http://205.174.165.80/CICDataset/ISCX-SlowDoS-2016/Dataset/). It was converted from pcap to flows using [flowtbag](https://github.com/DanielArndt/flowtbag). If you have questions on how to prepare the dataset check our pcap parser [repo](https://github.com/jatj/pcapParser)

# Dataset info:
Contains 24 h of network traffic with total size of 4.6 GB.

## Classes:
- slowbody2 (4 attacks)
- slowread (2 attacks)
- ddossim (2 attacks)
- goldeneye (3 attacks)
- slowheaders (5 attacks)
- rudy (4 attacks)
- hulk (4 attacks)
- slowloris (2 attacks)
