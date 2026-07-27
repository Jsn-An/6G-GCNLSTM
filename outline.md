# GNN-based Congestion Prediction and Task Offloading Optimization for Intelligent 5G/6G Edge Networks

## Project Overview

### Project Title

**GNN-based Congestion Prediction and Task Offloading Optimization for Intelligent 5G/6G Edge Networks**

中文：

**基于图神经网络的智能5G/6G边缘网络拥塞预测与任务卸载优化**

---

# 1. Research Background and Motivation

## 1.1 Background

With the development of 6G and AIoT (Artificial Intelligence of Things), future communication networks will connect massive intelligent devices, including:

- Autonomous vehicles
- Smart cameras
- Industrial robots
- IoT sensors
- Edge AI devices

These devices continuously generate large amounts of data, which need to be transmitted and processed through wireless networks and edge computing infrastructures.

A typical AIoT data transmission path is:

```
IoT Device
      |
      ↓
5G/6G Base Station
      |
      ↓
Edge Computing Server
      |
      ↓
Cloud/Data Center
```

With increasing device density and traffic demand, future networks may suffer from:

- Traffic overload
- Resource imbalance
- Network congestion
- Increased latency


Traditional network management methods rely on static configuration, which cannot efficiently handle dynamic and large-scale AIoT networks.

Therefore, an intelligent network management framework is required to:

1. Predict future network conditions;
2. Detect potential congestion in advance;
3. Dynamically optimize task allocation and network resources.

---

# 2. Research Idea

## 2.1 Core Concept

The communication network is modeled as a graph:

\[
G=(V,E)
\]

where:

- **Nodes (V)** represent network entities:
  - Base stations
  - Edge servers

- **Edges (E)** represent communication relationships:
  - Wireless connections
  - Network links


The communication network is similar to a traffic system:

| Traffic Network | Communication Network |
|---|---|
| Road intersection | Base station / Edge node |
| Road | Communication link |
| Vehicle flow | Data traffic |
| Traffic jam | Network congestion |
| Traffic control | Network optimization |


Because network nodes influence each other, Graph Neural Networks (GNNs) are used to capture spatial dependencies.

Meanwhile, network states change over time, so LSTM is introduced to model temporal dependencies.

The overall framework:

```
Historical Network State

        |
        ↓

       LSTM

(Temporal Dependency Learning)

        |
        ↓

       GCN

(Spatial Relationship Learning)

        |
        ↓

Future Network Load Prediction

        |
        ↓

Congestion Detection

        |
        ↓

Task Offloading Optimization
```

---

# 3. Dataset

## 3.1 Dataset Selection

Selected Dataset:

**5G Network KPI Dataset**

---

## 3.2 Reason for Dataset Selection

The dataset is selected because:

1. It is directly related to 5G/6G communication scenarios;
2. It contains real network performance indicators;
3. It can represent the operation status of future intelligent edge networks.

Compared with traditional backbone network datasets:

- GÉANT Dataset:
  - Good graph structure
  - Less related to 6G AIoT scenarios

- 5G KPI Dataset:
  - Stronger connection with cellular networks
  - More suitable for AIoT edge network research

---

## 3.3 Dataset Features

Typical KPI features include:

| Feature | Description |
|-|-|
| Cell ID | Base station identifier |
| Traffic Volume | Network traffic amount |
| Number of Users | Connected devices |
| PRB Utilization | Radio resource utilization |
| Latency | Communication delay |
| Packet Loss | Transmission quality |

Example:

```
Time        Cell      Traffic    Users    PRB     Delay

10:00       BS1       50Mbps     100      40%     10ms
10:00       BS2       80Mbps     200      70%     20ms
10:00       BS3       90Mbps     300      90%     50ms
```

---

# 4. Research Objectives

The project contains two major objectives.

---

# Objective 1: Network Congestion Prediction

## Goal

Predict future network load and identify possible congestion before it happens.

Input:

```
Historical KPI sequence

(t-n, ..., t)
```

Output:

```
Future Network Load

(t+1)
```

Example:

Prediction result:

```
BS3:

Future PRB Utilization = 95%

Congestion = True
```

---

# Objective 2: Task Offloading Optimization

## Goal

Use prediction results to proactively adjust task allocation.

When a node is predicted to become overloaded:

```
Predicted Load > Threshold
```

the system searches for nearby low-load nodes and migrates tasks.

Example:

Before optimization:

```
IoT Device

      |
      ↓

    BS-A

      |
      ↓

Edge Server A

Load = 95%
```

After optimization:

```
IoT Device

      |
      ↓

    BS-B

      |
      ↓

Edge Server B

Load = 40%
```

---

# 5. Experimental Workflow

Overall pipeline:

```
Step 1:
Load 5G KPI Dataset

        ↓

Step 2:
Data Preprocessing

        ↓

Step 3:
Construct Network Graph

        ↓

Step 4:
Train GCN-LSTM Prediction Model

        ↓

Step 5:
Detect Future Congestion

        ↓

Step 6:
Apply Task Offloading Strategy

        ↓

Step 7:
Evaluate Performance Improvement
```

---

# 6. Data Processing

## 6.1 Feature Selection

Selected node features:

```
X =
[
Traffic,
User Number,
PRB Utilization,
Latency
]
```

Each base station is represented by a feature vector.

---

## 6.2 Graph Construction

Since KPI datasets may not directly provide topology information, construct graph manually.

## Method 1: Geographic Graph

If location information exists:

```
Distance < threshold

        ↓

Create Edge
```

Example:

```
BS1 -------- BS2
```

---

## Method 2: Correlation Graph

Calculate similarity between nodes.

If two base stations have similar traffic patterns:

```
High Correlation

        ↓

Create Edge
```

Final graph:

```
G = (V,E)
```

---

# 7. Prediction Model

## 7.1 Model Selection

Model:

**GCN + LSTM**

Reason:

- LSTM learns temporal changes of network states;
- GCN learns relationships between neighboring network nodes.

---

## 7.2 Model Architecture

```
Input:

Historical KPI Data

        |
        ↓

      LSTM

Temporal Feature Extraction

        |
        ↓

       GCN

Spatial Feature Aggregation

        |
        ↓

 Fully Connected Layer

        |
        ↓

Future KPI Prediction
```

---

## 7.3 Prediction Target

Recommended prediction target:

Primary:

```
Future PRB Utilization
```

or:

```
Future Network Load
```

Congestion rule:

```
If predicted load > 90%

→ Congestion
```

---

# 8. Task Offloading Optimization

## 8.1 Optimization Strategy

Use:

**Prediction-guided Task Offloading**

No reinforcement learning is required.

A simple heuristic strategy is sufficient.

---

## 8.2 Algorithm

Pseudo-code:

```python
for each network node:

    if predicted_load > threshold:

        find neighboring node
        with minimum load

        migrate tasks
```

---

## 8.3 Example

Before:

```
BS1: 95%
BS2: 40%
BS3: 50%
```

Optimization:

```
Move tasks:

BS1 → BS2
```

After:

```
BS1: 70%
BS2: 65%
BS3: 50%
```

---

# 9. Experimental Evaluation

## 9.1 Prediction Performance

Compare different models:

### Baselines

1. LSTM
2. GCN
3. GCN-LSTM


Evaluation Metrics:

- MAE
- RMSE
- MAPE


Expected result:

GCN-LSTM performs better because it considers both:

- Temporal dependency
- Spatial dependency

---

# 9.2 Optimization Performance

Compare:

## Without Optimization

Only prediction.

## With Optimization

Prediction + Task Offloading.


Evaluation Metrics:

### 1. Maximum Node Utilization

Example:

Before:

```
95%
```

After:

```
70%
```

---

### 2. Number of Congested Nodes

Before:

```
5 nodes
```

After:

```
1 node
```

---

### 3. Average Latency

Before:

```
50 ms
```

After:

```
30 ms
```

---

# 10. Expected Contribution

This project demonstrates:

1. Modeling 5G/6G communication networks using graph structures;
2. Applying GNN to capture relationships between network nodes;
3. Predicting future congestion using GCN-LSTM;
4. Using prediction results to guide proactive task offloading;
5. Building an intelligent network management framework for future AIoT systems.

---

# 11. Project Scope

## Included

- 5G KPI Dataset
- Data preprocessing
- Network graph construction
- GCN-LSTM prediction
- Congestion detection
- Rule-based task offloading


## Not Included

- Reinforcement Learning
- Complex network simulator
- Digital Twin Network
- Advanced Dynamic GNN

---

# 12. Final Project Framework

```
              5G KPI Dataset

                     |

                     ↓

          Network Graph Construction

                     |

                     ↓

                 GCN + LSTM

                     |

                     ↓

          Future Load Prediction

                     |

                     ↓

          Congestion Detection

                     |

                     ↓

             Task Offloading

                     |

                     ↓

        Network Load Balancing
```