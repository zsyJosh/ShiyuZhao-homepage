---
title: "FOCUS: Fairness via Agent-Awareness for Federated Learning on Heterogeneous Data"
summary: "New metrics for fair Federated Learning for clients with heterogeneous data. Clustering improves fairness for Federated Learning."
tags:
- AI
date: "2022-10-25T18:50:20Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

links:
- name: arxiv
  url: https://arxiv.org/abs/2207.10265

url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

image:
  caption: 'An overview of our FOCUS algorithm'
  focal_point: Right

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example
---

Federated learning (FL) provides an effective collaborative training paradigm, allowing local agents to train a global model jointly without sharing their local data to protect privacy. However, due to the heterogeneous nature of local data, it is challenging to optimize or even define the fairness of the trained global model for the agents. For instance, existing work usually considers accuracy equity as fairness for different agents in FL, which is limited, especially under the heterogeneous setting, since it is intuitively "unfair" to enforce agents with high-quality data to achieve similar accuracy to those who contribute low-quality data. In this work, we aim to address such limitations and propose a formal fairness definition in FL, fairness via agent-awareness (FAA), which takes different contributions of heterogeneous agents into account. Under FAA, the performance of agents with high-quality data will not be sacrificed just due to the existence of large amounts of agents with low-quality data. In addition, we propose a fair FL training algorithm based on agent clustering (FOCUS) to achieve fairness in FL measured by FAA. Theoretically, we prove the convergence and optimality of FOCUS under mild conditions for linear and general convex loss functions with bounded smoothness. We also prove that FOCUS always achieves higher fairness in terms of FAA compared with standard FedAvg under both linear and general convex loss functions. Empirically, we evaluate FOCUS on four datasets, including synthetic data, images, and texts under different settings, and we show that FOCUS achieves significantly higher fairness in terms of FAA while maintaining similar or even higher prediction accuracy compared with FedAvg and other existing fair FL algorithms.     
