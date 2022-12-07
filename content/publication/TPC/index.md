---
title: "TPC: Transformation-Specific Smoothing for Point Cloud Models"

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here 
# and it will be replaced with their full name and linked to their profile.
authors:
- Wenda Chu
- Linyi Li
- Bo Li

# Author notes (optional)
# author_notes:
# - "Equal contribution"
# - "Equal contribution"

date: "2022-01-28T00:00:00Z"
doi: ""

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
publication: In *International Conference on Machine Learning*
publication_short: In *ICML* 2022

abstract: Point cloud models with neural network architectures have achieved great success and been widely used in safety-critical applications, such as Lidar-based recognition systems in autonomous vehicles. However, such models are shown vulnerable against adversarial attacks which aim to apply stealthy semantic transformations such as rotation and tapering to mislead model predictions. In this paper, we propose a transformation-specific smoothing framework TPC, which provides tight and scalable robustness guarantees for point cloud models against semantic transformation attacks. We first categorize common 3D transformations into additive (e.g., shearing), composable (e.g., rotation), and indirectly composable (e.g., tapering), and we present generic robustness certification strategies for all categories respectively. We then specify unique certification protocols for a range of specific semantic transformations and their compositions. Extensive experiments on several common 3D transformations show that TPC significantly outperforms the state of the art. For example, our framework boosts the certified accuracy against twisting transformation along z-axis within 20 degree from 20.3% to 83.8%.

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
url_pdf: ''
url_code: 'https://github.com/Qianhewu/Point-Cloud-Smoothing'
url_dataset: ''
url_poster: 'relref "../../pdf/poster/tpc_poster.pdf'
url_project: ''
url_slides: 'relref "../../pdf/slides/tpc_slides.pdf'
url_source: ''
url_video: 'relref "../../events/TPC_talk'


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  Placement: 1
  caption: 'Overview of TPC framework. TPC includes smoothing and certification strategies to provide certified robustness for point cloud models against semantic transformations. Besides rotation as shown in figure, TPC provides strong robustness certification for a wide range of other semantic transformations.'
  focal_point: "Center"
  preview_only: false
  alt_text: Overview of TPC framework.

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
