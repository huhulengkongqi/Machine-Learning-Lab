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