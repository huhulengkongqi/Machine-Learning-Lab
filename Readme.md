# README for Bin Packing Problem (BPP) Solution

This project provides solutions to three tasks related to the **Bin Packing Problem (BPP)** and its application in e-commerce packaging. The implementation includes training a policy network, using neural-guided search algorithms, and solving a real-world e-commerce packaging problem.

---

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Files Description](#files-description)
4. [Implementation Details](#implementation-details)
   - [Task 1: Policy Learning for BPP](#task-1-policy-learning-for-bpp)
   - [Task 2: Neural Guided Search Algorithm](#task-2-neural-guided-search-algorithm)
   - [Task 3: E-commerce Packaging Problem](#task-3-e-commerce-packaging-problem)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Acknowledgments](#acknowledgments)

---

## Overview

The **Bin Packing Problem (BPP)** involves packing objects of varying sizes into containers to minimize wasted space. This project includes:
- **Task 1**: Training a policy network using reinforcement learning or supervised learning to solve the BPP.
- **Task 2**: Combining the trained policy with tree search algorithms to improve packing efficiency.
- **Task 3**: Solving a real-world e-commerce packaging problem using multiple container sizes.

---

## Requirements

Install the necessary Python libraries:
```bash
pip install torch numpy pandas matplotlib
```

---

## Files Description

1. **`task1.py`**
   - Contains the implementation for **Task 1**, including:
     - Data generation for the BPP problem.
     - Training a Transformer-based policy network using reinforcement learning (PPO).
     - Evaluation of the policy network.

2. **`task2_and_task3.py`**
   - Handles **Task 2** and **Task 3**, including:
     - Neural-guided beam search for BPP.
     - A heuristic-based solution for e-commerce packaging with multiple containers.

3. **`task3.csv`**
   - The input file for Task 3 containing test data for e-commerce packaging.

---

## Implementation Details

### Task 1: Policy Learning for BPP
- **Data Generation**: Uses a custom algorithm to split and rotate items randomly to simulate a realistic BPP dataset.
- **Policy Network**: Implements a Transformer-based network to predict the placement of items in a 3D container.
- **Training**:
  - Utilizes **Proximal Policy Optimization (PPO)** for reinforcement learning.
  - Trains the model to maximize container space utilization.

### Task 2: Neural Guided Search Algorithm
- **Search Algorithm**: Combines the trained policy network from Task 1 with a **beam search algorithm** to optimize item placement.
- **Key Features**:
  - Evaluates items based on neural network scores.
  - Prioritizes items to maximize packing efficiency.

### Task 3: E-commerce Packaging Problem
- **Multi-container Packing**: Adapts the solution to handle multiple containers of various sizes.
- **Algorithm**:
  - Sorts items using a neural-guided beam search.
  - Places items in containers while ensuring no overlaps and minimizing wasted space.
- **Efficiency Metric**: Evaluates the packing efficiency as the ratio of packed item volume to container volume.

---

## How to Run

### Step 1: Train the Policy Network (Task 1)
Run the following command:
```bash
python task1.py
```
- Generates a dataset:
  - The dataset simulates a realistic Bin Packing Problem by splitting and rotating items randomly.

- Trains the policy network using PPO:
  - The Proximal Policy Optimization (PPO) algorithm is used to optimize the network.
  - The network learns to maximize container space utilization by predicting the optimal placement of items.

- Evaluates and visualizes the packing results:
  - Evaluates the trained policy network using test datasets.
  - Visualizes the item placement in a 3D container to showcase the efficiency of the trained model.

---

### Step 2: Solve BPP Using Neural-Guided Search (Task 2)
Run the following command:
```bash
python task2_and_task3.py
```
- Executes the neural-guided search algorithm:
  -Combines the trained policy network from Task 1 with a beam search algorithm.
  -Uses neural network predictions to guide the search process, improving the efficiency of item placement.
  -Optimizes packing for single-container problems:

- Focuses on maximizing the utilization of a single container.
  -Prioritizes items based on their neural network scores and physical properties (e.g., volume and dimensions).


### Step 3: Solve E-commerce Packaging Problem (Task 3)

Ensure `task3.csv` is in the current directory. Run:
```bash
python task2_and_task3.py
```

- Processes multiple orders:
  - Reads test data from `task3.csv`, which contains multiple e-commerce orders.
  - Each order consists of several items characterized by attributes such as length, width, height, and quantity.

- Outputs packing details and efficiency metrics for each order:
  - Assigns each item in an order to the most suitable container from a predefined set of sizes.
  - Computes the packing efficiency as the ratio of the total volume of packed items to the total volume of containers used.

---

## Results

### Task 1
- **Average Space Utilization**: Achieved >90% efficiency on test datasets.
- **Visualization**: Demonstrates 3D item placement within the container, showing how the policy network optimizes packing.

### Task 2
- **Neural-Guided Search**:
  - Combines the trained policy network with beam search to enhance packing efficiency.
  - Achieved better results compared to heuristic-only approaches.

### Task 3
- **E-commerce Packaging**:
  - Efficiently packed items into multiple containers, optimizing space utilization.
  - Demonstrated high packing efficiency across all test orders.

---

## Acknowledgments

This project leverages:
- **Reinforcement Learning (PPO)**: To train the policy network for space optimization.
- **Transformer Architecture**: For sequence-to-sequence prediction and item placement.
- **Tree Search Algorithms**: To guide packing decisions and improve efficiency.
