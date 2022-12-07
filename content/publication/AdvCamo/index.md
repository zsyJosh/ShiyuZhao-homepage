---
title: "Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling"

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here 
# and it will be replaced with their full name and linked to their profile.
authors:
- Zhanhao Hu
- Wenda Chu
- Xiaopei Zhu
- Hui Zhang
- Bo Zhang
- Xiaolin Hu

# Author notes (optional)
author_notes:
- "Equal contribution"
- "Equal contribution"

date: "2022-11-21T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
# publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["3"]

# Publication name and optional abbreviated publication name.
# publication: Preprint. Under Review
# publication_short: 
publication: Under Review
publication_short: Under Review

abstract: Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. In this work, we aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects,  humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumble-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spin (TPS) to narrow the gap between digital and real-world objects. We printed the developed 3D texture pieces on fabric materials and tailored them into T-shirts and trousers. Experiments show high attack success rates of these clothes against multiple detectors.

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
$- name: arxiv
#  url: https://arxiv.org/abs/2201.12733
url_pdf: ''
# url_code: 'https://github.com/Qianhewu/Point-Cloud-Smoothing'
url_dataset: ''
# url_poster: 'staticref "poster/tpc_poster.pdf'
url_project: ''
# url_slides: 'staticref "slides/tpc_slides.pdf'
# url_source: ''
# url_video: 'relref "../../events/TPC_talk'


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
# image:
#  Placement: 1
#  caption: 'Overview of TPC framework. TPC includes smoothing and certification strategies to provide certified robustness for point cloud models against semantic transformations. Besides rotation as shown in figure, TPC provides strong robustness certification for a wide range of other semantic transformations.'
#  focal_point: "Center"
#  preview_only: false
#  alt_text: Overview of TPC framework.

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
