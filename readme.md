# SpecRL: Specification-Guided Dense Rewards for VLA Reinforcement Learning

<p align="center">
  <img src="assets/overview.png" alt="SpecRL Overview" width="100%">
</p>

Official implementation of the paper: **"SpecRL: Specification-Guided Dense Rewards for VLA Reinforcement Learning"**.

🚨 Notice: Coming Soon

The complete source code, training scripts, and grounding modules (privileged & visual) are currently being organized and will be released here in a few days. Stay tuned!

---

## 📖 Abstract

Reinforcement learning fine-tuning (RLFT) has emerged as a promising approach for adapting pretrained vision-language-action (VLA) models through task-directed interaction. However, many existing methods rely on sparse binary rewards, which provide limited guidance for policy optimization and thereby hinder learning performance. While recent approaches seek denser feedback via LLM-generated reward functions or vision-language model evaluations, they are often inefficient and the resulting rewards are often unreliable, potentially introducing errors that misguide policy optimization.

In this work, we propose **SpecRL**, a specification-guided framework designed to provide dense, interpretable online rewards for RLFT. The core idea is to formalize task requirements in Signal Temporal Logic (STL) and derive rewards from its explicit quantitative semantics. Specifically, SpecRL leverages an LLM-aided process to convert language instructions into STL specifications constituted by task-level atomic propositions. During execution, SpecRL evaluates proposition satisfaction using either a privileged-state grounding or a lightweight visual grounding module, supporting both simulation and real-world deployments. A runtime monitor then aggregates the quantitative values of propositions along the specification's temporal structure, translating fine-grained task progress into dense rewards. Extensive evaluations on the LIBERO benchmark show that privileged-state SpecRL improves the success rate over binary-reward RLFT by $3.3$ percentage points, while visual SpecRL surpasses the strongest visual baseline by $4.8$ percentage points with $1.43\times$ faster training. Real-world trajectory evaluations further demonstrate that its deterministic and interpretable design delivers more accurate reward feedback than baselines.

---

## 🛠️ Framework Architecture

The SpecRL pipeline comprises two core stages:
1. **Offline Instruction Parsing:** Translates natural-language instructions into structured JSON task representations encoding STL specifications composed of domain-specific atomic propositions.
2. **Online Grounding & Monitoring:** Evaluates task-level atomic propositions using either privileged simulator states or visual observations (via YOLOE instance segmentation and depth-based 3D reconstruction). A finite-state automaton (FSA) and arithmetic-geometric mean conjunctions aggregate fine-grained progress into continuous, dense reward streams for online policy fine-tuning.

---


## 🚀 Getting Started

### Prerequisites & Dependencies
* Python 3.11.14+
* PyTorch 2.6+
* Ubuntu 22.04.4 LTS
* Hardware: NVIDIA A100 (80GB) GPUs recommended for VLA training.



## 🤖 Real-World Deployment
SpecRL is compatible with physical robot setups. Our real-world experiments are deployed on a **Franka Research 3** robotic manipulator equipped with an Intel RealSense D455 global camera and a wrist-mounted ZED Mini camera.

