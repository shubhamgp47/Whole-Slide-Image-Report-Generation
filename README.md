# Vision-Language Models for Pathology Report Generation from Gigapixel Whole-Slide Images![F1](https://github.com/user-attachments/assets/b7155d30-f7a3-4be9-84bc-3a39ddcd41d5)
## Project Overview
This project systematically compares patch-level and slide-level encoding paradigms
for diagnostic report generation in computational pathology. We evaluate four configurations
on the REG2025 dataset containing 8,352 WSI and report pairs spanning seven organ
types with standardized CAP protocol reports. The baseline HistGen framework with
DINOv2 ViT-L feature extractor and learned hierarchical aggregation achieves BLEU-4 of
0.614, ROUGE-L of 0.684, and REGScore of 0.676. Replacing the feature extractor with
increasingly large foundation models shows modest gains, with UNI reaches BLEU-4 of 0.622
,ROUGE-L of 0.694, and REGScore of 0.682. UNI2 achieves BLEU-4 of 0.640, ROUGE-L
of 0.714, and REGScore of 0.698. TITAN, a slide-level foundation model with pretrained
aggregation, significantly outperforms all patch-level approaches with BLEU-4 of 0.643,
ROUGE-L of 0.760, and REGScore of 0.742 while reducing training time by 30%. These
results demonstrate that slide-level encoding with pretrained encoder offers substantial
advantages over learned patch-level aggregation for automated report generation
