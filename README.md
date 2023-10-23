# Predicting Well Interference in Shale Gas Exploration using GraphSAGE and GRU

## Introduction

Hydraulic fracturing is a crucial process in making shale gas exploration profitable. However, the resulting interconnected fracture networks often lead to well interference, impacting production forecasts. Deep learning has emerged as a powerful tool for accurate production forecasting, but traditional time-series models struggle with inter-sample connections. To address this, we turn to Graph Convolutional Networks (GCNs) which excel at capturing well interference.

GCNs process data on irregular and non-Euclidean domains. Inspired by spectral convolution, they perform message passing through layer-wise propagation, progressively capturing more complex K-hop information. However, traditional GCNs operate under transductive learning, limiting predictions to known nodes and lacking batch efficiency. 

In this study, we employ the Graph Sampling and Aggregation (GraphSAGE) method, which follows inductive learning principles. By training on the aggregator to extract features from neighboring nodes, GraphSAGE enables generalization of predictions to unseen nodes. This approach proves crucial in capturing well interference, making predictions for new wells at any location, and adeptly monitoring dynamic changes in producing wells.

## Methodology

We leverage GraphSAGE in conjunction with Gated Recurrent Unit (GRU) and time-distributed layers to capture spatial-temporal (ST) information. Our approach allows the adjacency matrix to vary over time, enhancing adaptability.

### Input Shape

- X: [batch(node), sequence, features]
- 𝐴djacency matrix: [batch(node), sequence, node number]

## Data Description

We focus on 2240 wells with complete features, clustered based on pads using DBSCAN:

- 308 subgraphs (6.38 wells/pad)
- 280 (1833 wells) for training
- 28 (133 wells) for testing

We form disjoint unions for full batch training.

## Results

The figure displays predicted production curves of 16 wells from different pads, using our model trained on the full batch. These curves effectively capture both trends and sudden changes in production.

We also investigate the similarity in production profiles, particularly for the 11th and 17th pads. This showcases the resemblance in their production profiles, suggesting connections within the production behavior in the same pad.

For more detailed information, please refer to the original paper.

---

This research demonstrates the potential of integrating GraphSAGE with GRU for predicting well interference in shale gas exploration. The methodology proves robust and adaptable, paving the way for more accurate and reliable production forecasts in the industry.
