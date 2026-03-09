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


<img width="729" height="909" alt="Flowchart (1)" src="https://github.com/user-attachments/assets/4a00c638-7b81-474d-a771-9453dde72c2d" />

## Results
| Configuration | Feature Extractor | Encoder | REG Score | BLEU-4 | ROUGE-L |
|---------------|-------------------|---------|-----------|--------|---------|
| **HistGen** (baseline) | HistGen DINOv2 ViT-L | **HistGen Encoder** | **0.676** | 0.614 | 0.684 |
| UNI1 | UNI | **HistGen Encoder** | 0.682 | 0.622 | 0.694 |
| UNI2 | UNI2 | **HistGen Encoder** | **0.698** | 0.640 | 0.714 |
| **TITAN** (best) | CONCH | **Frozen TITAN** | **0.742** | **0.643** | **0.760** |

![Model Evaluation](Results/Model%20Evaluation/all_models_overview.png)

![Improvement over baseline](Results/Improvement%20Heatmap/heatmap_improvement_over_baseline.png)


## Statistical Significance vs HistGen Baseline 

**Legend**:Mean = Model mean - Baseline mean; CI = 95% bootstrap confidence interval
(10,000 iterations); Cohen’s d interpretation: small (0.2-0.5), medium (0.5-0.8), large (>0.8);
p-value from paired t-test at α = 0.05

| Model  | Metric   | Baseline Mean±SD | Model Mean±SD | Mean | 95% CI          | p-value | Cohen's d     |
|--------|----------|------------------|---------------|------|-----------------|---------|---------------|
| **UNI** | BLEU-1  | 0.709±0.029     | 0.721±0.032  | +0.012 | [-0.026, +0.051] | 0.629  | 0.23 (small) |
|        | BLEU-2  | 0.672±0.029     | 0.683±0.032  | +0.011 | [-0.026, +0.050] | 0.651  | 0.22 (small) |
|        | BLEU-3  | 0.640±0.029     | 0.650±0.030  | +0.009 | [-0.026, +0.047] | 0.677  | 0.20 (small) |
|        | BLEU-4  | 0.614±0.028     | 0.622±0.029  | +0.008 | [-0.026, +0.042] | 0.711  | 0.18 (small) |
|        | METEOR  | 0.443±0.016     | 0.450±0.020  | +0.007 | [-0.016, +0.030] | 0.625  | 0.24 (small) |
|        | ROUGE-L | 0.684±0.028     | 0.694±0.022  | +0.010 | [-0.020, +0.040] | 0.616  | 0.24 (small) |
|        | REGScore| 0.676±0.025     | 0.682±0.019  | +0.005 | [-0.020, +0.032] | 0.740  | 0.24 (small) |
| **UNI2** | BLEU-1 | 0.709±0.029  | 0.740±0.028  | +0.031 | [-0.007, +0.065] | 0.201  | 0.68 (medium)|
|        | BLEU-2  | 0.672±0.029     | 0.702±0.029  | +0.029 | [-0.011, +0.065] | 0.247  | 0.61 (medium)|
|        | BLEU-3  | 0.640±0.029     | 0.668±0.030  | +0.027 | [-0.011, +0.064] | 0.286  | 0.55 (medium)|
|        | BLEU-4  | 0.614±0.028     | 0.640±0.031  | +0.025 | [-0.015, +0.064] | 0.326  | 0.50 (medium)|
|        | METEOR  | 0.443±0.016     | 0.464±0.020  | +0.020 | [-0.003, +0.042] | 0.201  | 0.68 (medium)|
|        | ROUGE-L | 0.684±0.028     | 0.714±0.026  | +0.030 | [-0.007, +0.066] | 0.223  | 0.64 (medium)|
|        | REGScore| 0.676±0.025    | 0.698±0.027  | +0.022 | [-0.010, +0.055] | 0.322  | 0.86 (large) |
| **TITAN** | BLEU-1 | 0.709±0.029    | 0.739±0.043  | +0.030 | [-0.026, +0.072] | 0.347  | 0.48 (small) |
|        | BLEU-2  | 0.672±0.029     | 0.703±0.041  | +0.031 | [-0.024, +0.071] | 0.347  | 0.50 (medium)|
|        | BLEU-3  | 0.640±0.029     | 0.670±0.038  | +0.030 | [-0.021, +0.068] | 0.313  | 0.52 (medium)|
|        | BLEU-4  | 0.614±0.028     | 0.643±0.034  | +0.029 | [-0.018, +0.064] | 0.304  | 0.53 (medium)|
|        | METEOR  | 0.443±0.016     | 0.467±0.022  | +0.023 | [-0.006, +0.048] | 0.207  | 0.67 (medium)|
|        | **ROUGE-L** | 0.684±0.028 | 0.760±0.025 | +0.076 | [+0.033, +0.108] | **0.027** | **1.53 (large)** |
|        | **REGScore** | 0.676±0.025 | 0.742±0.023 | +0.066 | [+0.030, +0.095] | **0.025** | **2.81 (large)** |


## Usage
1. Preprocessing
   
   For segnmentation use [CLAM patching script](/HistGen/CLAM/patching_scripts/tcga-wsi-report.sh) using the clam conda environment.
   
   For feature extraction using HistGen feature extractor, Uni or Uni2 use the respective files in [Feature extraction](/HistGen/CLAM/extract_scripts) using clam conda environment. For CONCHv1.5 use the [Feature extraction](/HistGen4TITAN/CONCH%20CLAM/extract_features_calling_script.sh) and post process the features with [Postprocessing for CONCHv1.5](/HistGen4TITAN/CONCH%20CLAM/PostProcess%20CONCH%20Features/postprocess_featues.ipynb) using clam_conch environment.
   
   To create slide embeddings ftom TITAN use [TITAN Slide Embeddings](/HistGen4TITAN/extractSlideEmbeddings.py) using histgen_titan environment.

2. Training
   
   To train Histgen baseline and other patch level encoder variants (UNI and UNI2) use the files [HistGen Training](/HistGen/train_wsi_reportseed4x.sh), [UNI Training](/HistGen/train_wsi_report_uni1_seed4x.sh), and [UNI2 Training](/HistGen/train_wsi_report_uni2_seed4x.sh) respectively using the histgen environment.

   To train TITAN, use [TITAN Training](/HistGen4TITAN/train_wsi_report_TITAN.sh) using histgen_titan environment.

4. Inference

   For BLUE, METEOR, ROUGE-L, use the files [HistGen Testing](/HistGen/test_wsi_report_seed4x.sh), [UNI Testing](/HistGen/test_wsi_report_UNI1_seed4x.sh) [UNI2 Testing](/HistGen/test_wsi_report_UNI2_1_seed4x.sh) with histgen environment. For TITAN use [TITAN Testing](/HistGen4TITAN/test_wsi_report_5_seed4x.sh) with histgen_titan environment.

   For REGScore use the file [HistGen Testing](/Other%20Activities/REG2025%20Inference/reg_evaluator.py) with reg2025-eval environmet for all the models.

Sample job scripts are present in [Scripts](/Job%20Scripts/)

