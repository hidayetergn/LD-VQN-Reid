# LD-VQN-Reid
Linguistically Driven Visual Query Network (LD-VQN)

├── configs/                     # Eğitim ve test konfigürasyonları (Hyperparameters)
│   ├── veri776_ldvqn.yml
│   └── vehicleid_ldvqn.yml
├── datasets/                    # Veri yükleyiciler ve 417K JSONL formatlı metin verileri
│   ├── build_dataloader.py
│   ├── veri776_dataset.py
│   └── vehicleid_dataset.py
├── models/                      # LD-VQN mimarisi ve çekirdek modüller
│   ├── ld_vqn.py                # Ana model (MLA + D-TCR birleşimi)
│   ├── mla.py                   # Multi-Level Adapter
│   ├── d_tcr.py                 # Dynamic Text-Conditioned Routing
│   ├── losses/                  # Ortogonal Disentanglement ve Triplet Loss
│   │   ├── orthogonal_loss.py
│   │   └── xbm_triplet.py
│   └── text_encoders/           # DeBERTa entegrasyonu
├── scripts/                     # Tek tıkla çalıştırma betikleri
│   ├── train_veri776.sh
│   ├── eval_vehicleid.sh
│   └── extract_attention.sh
├── tools/                       # Görselleştirme araçları (Makaledeki figürler için)
│   ├── visualize_attention.py   # Fig. 9: Cross-modal attention haritaları
│   └── tsne_visualization.py    # Fig. 11: t-SNE kümeleme görselleştirmesi
├── weights/                     # Pretrained model ağırlıklarının konulacağı dizin
├── Dockerfile                   # Çevre kurulumu için kusursuz container
├── environment.yml              # Conda kurulumu için
├── requirements.txt             # Pip kurulumu için
├── train.py                     # Ana eğitim döngüsü (Algorithm 1)
├── test.py                      # Değerlendirme ve mAP/Rank hesaplama
└── README.md              


# Breaking the Semantic Bottleneck: Linguistically Driven Visual Query Networks (LD-VQN)

[![Paper](https://img.shields.io/badge/Paper-Anonymous%20Submission-blue.svg)](link_to_pdf)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Note to Reviewers:** This repository is strictly anonymized for the double-blind review process of TPAMI. The full dataset (417K textual annotations) and pre-trained weights have been provided via anonymous Google Drive links to facilitate full reproducibility.

Official PyTorch implementation of **LD-VQN**, a novel cross-modal architecture for fine-grained Vehicle Re-Identification. We introduce the first large-scale multimodal benchmark featuring over 417,000 unconstrained natural language descriptions to resolve the "semantic bottleneck" in current Vision-Language Models (VLMs).

## 🚀 Highlights / TL;DR
- **417K Multimodal Corpus:** We semantically enriched VeRi-776 and VehicleID datasets with high-density textual descriptions, bridging the gap between macroscopic shapes and instance-level primitives (e.g., custom decals, specific grille geometries).
- **Multi-Level Adapter (MLA):** Rescues early-stage textural primitives from the vision backbone before they homogenize.
- **Dynamic Text-Conditioned Routing (D-TCR):** Dynamically infers spatial adjacency directly from text, routing specific linguistic tokens to exact geometric coordinates.
- **Orthogonal Disentanglement Loss:** Mathematically forces latent semantic nodes toward a $\pi/2$ separation, strictly preventing attention collapse.

<div align="center">
  <img src="https://via.placeholder.com/800x400?text=Insert+Figure+6+(Overall+Architecture)+Here" alt="LD-VQN Architecture">
  <p><em>Figure 1: Overall architecture of the proposed LD-VQN.</em></p>
</div>

---

## 🛠️ Environment Setup & Installation

We provide both Conda and Docker setups to ensure 100% reproducibility.

### Option 1: Conda Environment (Recommended)
```bash
git clone [https://github.com/LDVQN-Paper/LD-VQN.git](https://github.com/LDVQN-Paper/LD-VQN.git)
cd LD-VQN
conda env create -f environment.yml
conda activate ldvqn_env
