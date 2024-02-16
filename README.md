# Infinity: A Unified Solution of General Intelligence

This implementation is a **proof-of-concept**. It has all the functionalities that constitute a true system of general intelligence. The only disadvantage is the fact that it **does not support parallel execution**. Therefore, the main purpose is to provide a testing playground and help in understanding the behavior of this type of network.

## [Whitepaper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4726421)

**Abstract**. We start by proposing a fundamental principle of general intelligence. We then propose a special type of network precisely built around that principle. The network’s unified architecture employs a Hebbian-like prediction-based learning strategy. Predictions are the key trigger events that control every action in the network. The network is a sequence memory capable of identifying high-order spatial and temporal patterns. It elegantly solves the challenges of short-to-long term memory and working memory. Minimization is enforced on every level. At any point in time, the total number of elements in the network and their activity will be minimal. This enables high performance and general efficiency. By combining the power of combinatorics with sparse coding and population coding, astronomically large number of unique patterns can be represented. Long term potentiation and depression enable the network to directly control the expression of every single pattern individually. Thinking is a special emergent force in the network. Motor output is generated directly from the network’s internal model.

## Get started
* Read the  [whitepaper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4726421)
* Clone repo `git clone git@github.com:rogic89/infinity.git`
* Download and unzip initial [datasets](https://drive.proton.me/urls/K4AQJWV0KG#ilEPSrzAR3ML) into the `infinity` folder
* Navigate into  `src` folder
* Run network `node run`
* Check terminal for logs

## Datasets
Inside the `src/run.js` is a list of initial datasets.
```
const datasource = '../datasets/mnist';
// const datasource = '../datasets/emnist-mnist';
// const datasource = '../datasets/emnist-fashion';
// const datasource = '../datasets/emnist-letters';
// const datasource = '../datasets/emnist-balanced';
```
Simply uncomment a line to run a different dataset.

## Log
While the network is running, it will print a log after each timestep. Each log contains the current state of the network and all of its elements. By carefully analyzing the logs, timestep after timestep, it is possible to understand the entire learning process of the network.
```
/////////////////////////////////////// TIMESTEP 60000
/////////////////////////////////////// LABEL 1
================ NETWORK ==============
    Output/Average -> 6.62%
    Unpred/Average -> 0.17%
  Thinking/Average -> 0%
  Ncreated/Average -> 95.02%
   Ncreated/Unpred -> 1 - 0%
================ NODES ================ LAYER A2
            Output -> 30 - 156,157,159,185,187,212,241,243,268,297,322,324,353,378,379,407,409,434,462,463,488,518,519,544,574,575,600,628,629,631
       Unpredicted -> 0
          Thinking -> 0
   Output/Timestep -> 3.83%
================ LINKS ================
              Live -> 13.12k
           Deleted -> 31.71k - 70.73%
  Created/Timestep -> 0
  Deleted/Timestep -> 0
        Links/Pool -> 6.12
        Links/Node -> 16.74
          Sparsity -> 2.14%
 HighestPermanence -> Maximum
================ POOLS ================
              Live -> 2.14k
           Deleted -> 2.94k - 57.86%
  Created/Timestep -> 0
  Deleted/Timestep -> 0
        Pools/Node -> 2.73
          Sparsity -> 0.7%
    PermanentPools -> 2.04k - 95.2%
```

## Score
After the network has completed processing the dataset, it will generate a score. The score does not accurately describe the performance or functionality of this type of network. The network learns as humans do. It observes temporal sequences and strengthens patterns that repeat many times in quick succession. Therefore, the datasets should be **sorted** by labels otherwise it may take longer for the network to stabilize. Learning is completely unsupervised and we are only trying to understand the inner workings of the network.
```
/////////////////////////////////////// DATASET
/////////////////////////////////////// SCORE
0 100
1 99
2 93
3 97
4 94
5 91
6 95
7 97
8 92
9 99
SCORE: 95.7
TIME-TOTAL: 29.957s
