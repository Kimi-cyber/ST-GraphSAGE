# Shale Gas Production Forecasting with Well Interference Based on Spatial-Temporal Graph Convolution Network

## Citation

If you use this code in your research, please cite the following paper: [Xu, Z., & Leung, J. Y. (2023, October). Shale Gas Production Forecasting with Well Interference Based on Spatial-Temporal Graph Convolutional Network. In SPE Annual Technical Conference and Exhibition(p. D031S032R004). SPE.](https://doi.org/10.2118/215056-MS)

## Introduction

Hydraulic fracturing is a crucial process in making shale gas exploration profitable. However, the resulting interconnected fracture networks often lead to well interference, impacting production forecasts. Deep learning has emerged as a powerful tool for accurate production forecasting, but traditional time-series models struggle with inter-sample connections. To address this, we turn to Graph Convolutional Networks (GCNs) which excel at capturing well interference.

GCNs process data on irregular and non-Euclidean domains. Inspired by spectral convolution, they perform message passing through layer-wise propagation, progressively capturing more complex K-hop information. However, traditional GCNs operate under transductive learning, limiting predictions to known nodes and lacking batch efficiency. 

In this study, we employ the Graph Sampling and Aggregation (GraphSAGE) method, which follows inductive learning principles. By training on the aggregator to extract features from neighboring nodes, GraphSAGE enables the generalization of predictions to unseen nodes. This approach proves crucial in capturing well interference, making predictions for new wells at any location, and adeptly monitoring dynamic changes in producing wells.

<div style="text-align:center">
  <img src="https://github.com/ziming-zx/ST-GraphSAGE/assets/55851734/ec4b0f24-a93e-4c31-8501-15b8d645389d" alt="GitHub Logo" width="500">
</div>
https://dnicolasespinoza.github.io/node57.html 

## Methodology

We leverage GraphSAGE in conjunction with Gated Recurrent Unit (GRU) and time-distributed layers to capture spatial-temporal (ST) information. Our approach allows the adjacency matrix to vary over time, enhancing adaptability.

![image](https://github.com/ziming-zx/ST-GraphSAGE/assets/55851734/7585cf40-6598-40ac-b95f-fc510a0e4f78)



### Input Shape

- X: [batch(node), sequence, features]
- 𝐴djacency matrix: [batch(node), sequence, node number]

## Data Description

We focus on 2240 wells with complete features, clustered based on pads using DBSCAN:

- 308 subgraphs (6.38 wells/pad)
- 280 (1833 wells) for training
- 28 (133 wells) for testing

We form disjoint unions for full batch training.

![image](https://github.com/ziming-zx/ST-GraphSAGE/assets/55851734/05326eaa-2be4-4d85-af6b-b15faa632392)


## Results

The task involves utilizing a 12-month historical dataset and features such as reservoir properties and well completion information to forecast the subsequent 36 months. The figure displays predicted production curves of 16 wells from different pads, using our model trained on the full batch. These curves effectively capture both trends and sudden changes in production.

![image](https://github.com/ziming-zx/ST-GraphSAGE/assets/55851734/10b324aa-7f64-4e0f-9218-c1726a475bc1)


We also investigate the similarity in production profiles, particularly for the 11th and 17th pads. This showcases the resemblance in their production profiles, suggesting connections within the production behavior in the same pad.

![image](https://github.com/ziming-zx/ST-GraphSAGE/assets/55851734/db352245-7383-4449-851e-65b18081ae06)

![image](https://github.com/ziming-zx/ST-GraphSAGE/assets/55851734/0f3f75af-4524-43ea-be6b-954bee4d2c7b)


For more detailed information, please refer to the original paper.

---

This research demonstrates the potential of integrating GraphSAGE with Recurrent Neural Networks for predicting well interference in shale gas exploration. The methodology proves robust and adaptable, paving the way for more accurate and reliable production forecasts in the industry.
