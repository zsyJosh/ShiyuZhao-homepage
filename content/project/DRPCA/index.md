---
title: Distributed Robust Principal Component Analysis
summary: A distributed robust principal component analysis algorithm with some convergence guarantees.
tags:
- Others
date: "2022-07-02T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

links:
- name: arxiv
  url: https://arxiv.org/abs/2207.11669

url_code: "https://github.com/Qianhewu/Distributed-RPCA"
url_pdf: ""
url_slides: ""
url_video: ""

image:
  caption: 'Relative recover error of different problem scales.'
  focal_point: Right

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example
---

Principal component analysis (PCA) has been widely used for dimension reduction in data science. It extracts the top k significant components of a given matrix by computing the best low-rank approximation. However, it is well known that PCA is sensitive to noises and adversarial attacks. Robust PCA (RPCA) aims at mitigating this drawback by separating the noise out explicitly. Specifically, RPCA assumes that the observed matrix $M$ can be decomposed as $M = L^* + S^*$ where $L^*$ is a low-rank matrix and $S^*$ is a sparse matrix.

Some RPCA algorithms relax the low-rank constraints to nuclear norm and sparsity to $\ell_1$ norm, so that traditional convex optimization algorithms (e.g., PGM, ADMM) can be directly applied. Others reformulate the problem as low-rank matrix factorization with $\ell_1$ norm bounded noise. However, none of these algorithms are scalable and can be implemented distributedly, due to the use of SVD or full matrix multiplications. In this paper, we propose a distributed RPCA algorithm based on consensus-factorization (DCF-PCA) that takes $O(1)$ computation time as the number of remote clients increase. We show the convergence of our algorithm both theoretically and empirically.
