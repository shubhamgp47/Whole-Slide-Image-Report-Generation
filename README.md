# Vision-Language Models for Pathology Report Generation from Gigapixel Whole-Slide Images![F1](https://github.com/user-attachments/assets/b7155d30-f7a3-4be9-84bc-3a39ddcd41d5)
## Aim of the Project
This project systematically compares patch-level and slide-level encoding paradigms
for diagnostic report generation in computational pathology. We evaluate four configurations
on the REG2025 dataset containing 8,352 WSI and report pairs spanning seven organ
types with standardized CAP protocol reports. We use the Histgen codebase aa baseline and improve it further. We aim to answer the following research questions -

RQ1: Does slide-level feature encoding with pretrained aggregation outperform
patch-level encoding with learned hierarchical aggregation for generating pathology
reports from WSIs?

RQ2: How does the scale of foundation model pretraining (model capacity and corpus
size) affect downstream report generation performance?

RQ3: Does a multimodal vision-language foundation model with slide-level encoding
outperform vision-only patch-level foundation models for automated report generation?

RQ4: What architectural and optimization adaptations are required when fine tuning
HistGen framework on frozen pretrained foundation model used as feature extractor?

## Results
| Configuration | Feature Extractor | Encoder | REG Score | BLEU-4 | ROUGE-L |
|---------------|-------------------|---------|-----------|--------|---------|
| **HistGen** (baseline) | HistGen DINOv2 ViT-L | **HistGen Encoder** | **0.676** | 0.614 | 0.684 |
| UNI1 | UNI | **HistGen Encoder** | 0.682 | 0.622 | 0.694 |
| UNI2 | UNI2 | **HistGen Encoder** | **0.698** | 0.640 | 0.714 |
| **TITAN** (best) | CONCH | **Frozen TITAN** | **0.742** | **0.643** | **0.760** |


