# Invoice AI — Relatório Final

## Resumo

Pipeline end-to-end para extração automática de dados estruturados de recibos, combinando técnicas de pré-processamento de imagem, OCR (Optical Character Recognition), classificação de qualidade via CNN, e parsing inteligente (Regex + LLM).

**Resultados principais:**

- Preprocessing reduz CER em 46%
- PaddleOCR supera Tesseract em 4x
- CNN Quality Gate com threshold 0.3 reduz CER em 43%
- LLM (Gemini 3.5 Flash) supera Regex em +23% F1

---

## 1. Introdução

### 1.1 Problema

A extração automática de dados de recibos (notas fiscais, cupons) é um problema recorrente em fintechs, contabilidade e gestão de despesas. O processo manual é lento, caro e propenso a erros.

### 1.2 Objetivo

Construir um pipeline que receba uma **foto de recibo** e retorne um **JSON estruturado** com itens, preços, total, impostos — avaliando rigorosamente cada componente.

### 1.3 Dataset

- **CORD-v2** (Consolidated Receipt Dataset): 1000 recibos com ground truth
- Splits: train (800), validation (100), test (100)
- Ground truth: JSON com menu items, subtotal, total, tax

---

## 2. Arquitetura do Pipeline

```
Imagem → Preprocessing → Quality Gate (CNN) → OCR → Postprocessing → Parser → JSON
```

### 2.1 Preprocessing

| Técnica            | Implementação                                    | Impacto               |
| ------------------ | ------------------------------------------------ | --------------------- |
| Document Detection | Contour + Perspective Transform                  | -37% CER              |
| Deskew             | Hough Transform / minAreaRect                    | Sem efeito no CORD-v2 |
| CLAHE              | Contrast Limited Adaptive Histogram Equalization | -38% CER              |
| Bilateral Filter   | Edge-preserving denoising                        | Estabiliza CLAHE      |
| Adaptive Threshold | Binarização adaptativa                           | ⚠️ PIORA +147%        |

### 2.2 Quality Gate (CNN)

- **Modelo:** MobileNetV2 com transfer learning (ImageNet)
- **Classes:** ready (CER<5%), marginal (5-20%), not_ready (>20%)
- **Dataset sintético:** 3791 imagens com degradações (blur, noise, JPEG, rotation)
- **Treino:** 2 fases (backbone congelado → fine-tune)
- **Resultado:** Test accuracy 70.4%, weighted F1 = 0.709

### 2.3 OCR

| Engine        | CER       | WER       | Latência |
| ------------- | --------- | --------- | -------- |
| **PaddleOCR** | **0.158** | **0.259** | ~12s     |
| Tesseract     | 0.667     | 1.155     | ~0.7s    |

PaddleOCR escolhido: 4x melhor qualidade, apesar de mais lento.

### 2.4 Parsing

| Parser                     | F1        | ANLS      | Latência | Custo      |
| -------------------------- | --------- | --------- | -------- | ---------- |
| Regex                      | 0.554     | 0.517     | 0.001ms  | $0         |
| **LLM (Gemini 3.5 Flash)** | **0.681** | **0.747** | ~3s      | ~$0.01/rec |

---

## 3. Experimentos

### 3.1 Metodologia

- 13 experimentos controlados
- 10 imagens do test set por experimento
- Métricas: CER, WER, F1, ANLS, Exact Match
- Cada experimento isola uma variável

### 3.2 Resultados

| #   | Experimento            | Resultado                                      |
| --- | ---------------------- | ---------------------------------------------- |
| 1   | Preprocessing ON/OFF   | CER: 0.294 → 0.158 (**-46%**)                  |
| 2   | CLAHE ON/OFF           | CER: 0.254 → 0.158 (**-38%**)                  |
| 3   | Adaptive Threshold     | CER: 0.158 → 0.390 (**+147%** ⚠️)              |
| 4   | Deskew ON/OFF          | Sem efeito (imagens já alinhadas)              |
| 5   | Ablation               | Doc Detection = maior ganho individual         |
| 6   | PaddleOCR vs Tesseract | PaddleOCR **4x melhor** (0.158 vs 0.667)       |
| 7   | CNN Filter             | 10% imagens rejeitadas                         |
| 8   | Threshold CNN          | **0.3 = sweet spot** (CER=0.090, 30% reject)   |
| 9   | Regex vs LLM           | LLM **+23% F1**, +44% ANLS                     |
| 10  | Zero-shot vs Few-shot  | **Idênticos** (F1=0.789)                       |
| 11  | GT Text vs Real OCR    | Real OCR ligeiramente melhor (parser adaptado) |
| 12  | Full Pipeline vs Naive | CER **-46%**, F1 **+26%**                      |
| 13  | Resoluções             | 25%: 0.615, 50%: 0.322, 100%: **0.158**        |

### 3.3 Key Findings

1. **Preprocessing é essencial** — reduz CER pela metade
2. **Document Detection** é a etapa mais impactante individualmente
3. **Adaptive Threshold prejudica** — destrói informação que o OCR precisa
4. **Resolução é crítica** — imagens em 25% são inutilizáveis
5. **PaddleOCR >> Tesseract** para recibos (4x melhor CER)
6. **CNN Quality Gate** a 0.3 é o melhor trade-off (economiza 30% de processamento)
7. **LLM é significativamente melhor** que Regex para parsing (+23% F1)
8. **Few-shot não ajuda** — modelos modernos entendem a tarefa sem exemplos

---

## 4. CNN de Qualidade — Detalhes

### 4.1 Geração de Dataset Sintético

- 632 imagens originais do CORD-v2 train
- 15 tipos de degradação (blur, noise, rotation, JPEG, etc.)
- 4632 imagens processadas com PaddleOCR
- CER medido → labels automáticos (ready/marginal/not_ready)
- Resultado: 688 ready (18%), 1156 marginal (31%), 1947 not_ready (51%)

### 4.2 Modelo

- MobileNetV2 (pretrained ImageNet)
- Custom head: Dropout(0.2) → Linear(1280, 3)
- Class weights para balanceamento
- Treino: 5 epochs frozen + 10 epochs fine-tune
- Device: Apple MPS (Metal Performance Shaders)

### 4.3 Resultados

| Classe           | Precision | Recall    | F1        |
| ---------------- | --------- | --------- | --------- |
| ready            | 0.617     | 0.760     | 0.681     |
| marginal         | 0.581     | 0.638     | 0.608     |
| not_ready        | 0.841     | 0.724     | 0.778     |
| **weighted avg** | **0.721** | **0.704** | **0.709** |

---

## 5. Stack Tecnológica

| Componente    | Tecnologia                               |
| ------------- | ---------------------------------------- |
| OCR           | PaddleOCR v6 (PP-OCRv6)                  |
| CNN           | PyTorch + MobileNetV2                    |
| LLM           | Google Gemini 3.5 Flash                  |
| Preprocessing | OpenCV (CLAHE, bilateral, morphological) |
| Parsing       | Pydantic schemas + Regex / Gemini        |
| Métricas      | editdistance, scikit-learn               |
| Visualização  | matplotlib, seaborn                      |
| Dataset       | CORD-v2                                  |

---

## 6. Limitações e Trabalhos Futuros

### Limitações

- Dataset pequeno (10 samples por experimento) — limita significância estatística
- Gemini free tier (20 req/dia) limitou testes do LLM parser
- Deskew sem efeito porque CORD-v2 já é alinhado
- CNN accuracy 70% — marginal vs not_ready é difícil de distinguir

### Trabalhos Futuros

- Rodar experimentos com 50+ samples (tier pago)
- Testar EasyOCR e TrOCR como alternativas
- Fine-tune do LLM com exemplos específicos de recibos brasileiros
- Deploy com FastAPI + Streamlit demo
- Aumentar dataset sintético com mais variedade de degradações

---

## 7. Conclusão

O pipeline desenvolvido demonstra que a combinação de:

1. **Preprocessing adaptado** (sem adaptive threshold, com CLAHE)
2. **Quality gate inteligente** (CNN rejeita imagens ruins)
3. **OCR de alta qualidade** (PaddleOCR)
4. **Parsing via LLM** (Gemini 3.5 Flash)

...produz resultados significativamente melhores que abordagens naive. O preprocessing sozinho reduz CER em 46%, e o LLM parser supera regex em 23% F1. A CNN quality gate oferece um trade-off elegante entre qualidade e eficiência, rejeitando imagens que gerariam resultados ruins.

---

## Apêndice: Figuras Geradas

| Figura               | Localização                                      |
| -------------------- | ------------------------------------------------ |
| CER por técnica      | `results/figures/final_cer_by_technique.png`     |
| CER boxplot          | `results/figures/final_cer_boxplot.png`          |
| F1 vs threshold      | `results/figures/final_f1_vs_threshold.png`      |
| Radar Regex vs LLM   | `results/figures/final_radar_regex_vs_llm.png`   |
| Waterfall CER        | `results/figures/final_waterfall_cer.png`        |
| Heatmap F1 por campo | `results/figures/final_heatmap_f1_by_field.png`  |
| Grid preprocessing   | `results/figures/final_preprocessing_grid.png`   |
| Pipeline visual      | `results/figures/report_pipeline_visual.png`     |
| CLAHE histograma     | `results/figures/report_clahe_histogram.png`     |
| Deskew antes/depois  | `results/figures/report_deskew_before_after.png` |
| OCR bounding boxes   | `results/figures/report_ocr_bboxes.png`          |
| Grad-CAM CNN         | `results/figures/report_gradcam.png`             |
| CNN architecture     | `results/figures/report_cnn_architecture.png`    |
| Failure example      | `results/figures/report_failure_example.png`     |
| Training curves      | `results/figures/quality_training_curves.png`    |
| Confusion matrix     | `results/figures/quality_confusion_matrix.png`   |
| ROC curve            | `results/figures/quality_roc_curve.png`          |
| Score vs CER         | `results/figures/quality_score_vs_cer.png`       |
