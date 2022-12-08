---
title: "Mask and Reason: Pre-Training Knowledge Graph Transformers for Complex Logical Queries"

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here 
# and it will be replaced with their full name and linked to their profile.
authors:
- Xiao Liu 
- Shiyu Zhao 
- Kai Su 
- Yukuo Cen 
- Jiezhong Qiu 
- Mengdi Zhang 
- Wei Wu 
- Yuxiao Dong 
- Jie Tang

# Author notes (optional)
# author_notes:
- "Equal contribution"
- "Equal contribution"
- "Equal contribution"

date: "2022-08-14T00:00:00Z"
doi: "https://doi.org/10.1145/3534678.3539472"

# Schedule page publish date (NOT publication's date).
# publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
# publication: Preprint. Under Review
# publication_short: 
publication: In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (Research Track)
publication_short: In *KDD* 2022

abstract: Knowledge graph (KG) embeddings have been a mainstream approach for reasoning over incomplete KGs. However, limited by their inherently shallow and static architectures, they can hardly deal with the rising focus on complex logical queries, which comprise logical operators, imputed edges, multiple source entities, and unknown intermediate entities. In this work, we present the Knowledge Graph Transformer (kgTransformer) with masked pre-training and fine-tuning strategies. We design a KG triple transformation method to enable Transformer to handle KGs, which is further strengthened by the Mixture-of-Experts (MoE) sparse activation. We then formulate the complex logical queries as masked prediction and introduce a two-stage masked pre-training strategy to improve transferability and generalizability. Extensive experiments on two benchmarks demonstrate that kgTransformer can consistently outperform both KG embedding-based baselines and advanced encoders on nine in-domain and out-of-domain reasoning tasks. Additionally, kgTransformer can reason with explainability via providing the full reasoning paths to interpret given answers. 
# Summary. An optional shortened abstract.
summary: 

tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org
links:
- name: arxiv
  url: https://arxiv.org/abs/2201.12733
url_pdf: 'relref "../../pdf/papers/kgtransformer.pdf'
url_code: 'https://github.com/THUDM/kgTransformer'
url_slides: 'relref "../../pdf/slides/kgtransformer_slides.pdf'



# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  Placement: 1
  caption: '(a) kgTransformer with Mixture-of-Experts is a high-capacity architecture
  that can capture EPFO queries with exponential complexity. (b) Two-stage pre-training trades off general knowledge and task-specific sparse
  property. Together with fine-tuning, kgTransformer can achieve better in-domain performance and out-of-domain generalization.'
  focal_point: "Center"
  preview_only: false
  alt_text: Overview of kgtransformer framework.

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
# - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example
---
