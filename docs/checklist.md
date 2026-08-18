# Invoice AI — Checklist de Implementação

> Marque com `[x]` as tarefas concluídas. Ex: `- [x] Tarefa feita`

---

## 🏗️ Semana 1: Fundação

### Setup do Projeto

- [x] Criar estrutura de pastas conforme definido no plano
- [x] Inicializar repositório Git + `.gitignore`
- [x] Criar `requirements.txt` / `pyproject.toml`
- [x] Criar `Makefile` com comandos básicos (`make train`, `make evaluate`, `make demo`)
- [x] Criar `README.md` inicial com descrição do projeto
- [x] Criar `configs/pipeline.yaml`, `preprocessing.yaml`, `experiments.yaml`

### Dataset (CORD-v2)

- [x] Fazer download do CORD-v2
- [x] Verificar formato das anotações vs schema Pydantic planejado
- [x] Notebook `01_eda.ipynb` — análise exploratória (distribuição de campos, tamanho de imagens, etc.)

### Pré-processamento

- [x] Implementar `src/preprocessing/document_detector.py` (contour detection + perspective transform)
- [x] Implementar `src/preprocessing/deskew.py` (Hough Transform / minAreaRect)
- [x] Implementar `src/preprocessing/enhancement.py` (CLAHE, Gaussian Blur, Adaptive Threshold)
- [x] Implementar `src/preprocessing/transforms.py` (Morphological Closing, Bilateral Filter)
- [x] Notebook `02_preprocessing.ipynb` — visualizar antes/depois de cada técnica

### OCR

- [x] Implementar `src/ocr/engine.py` (interface abstrata)
- [x] Implementar `src/ocr/paddle_ocr.py`
- [x] Implementar `src/ocr/tesseract_ocr.py`
- [x] Implementar `src/ocr/postprocessing.py` (correção O→0, l→1, merge de boxes)
- [x] Testar OCR em subset de imagens e validar output

---

## 🧠 Semana 2: CNN + Parsers

### CNN de Qualidade

- [x] Implementar script de geração de degradações sintéticas (blur, rotação, noise, JPEG, etc.)
- [x] Gerar dataset sintético (~5000 imagens: 1000 × 5 degradações)
- [x] Definir labels automáticos (CER < 5% → ready, > 20% → not_ready, entre → marginal)
- [x] Implementar `src/quality/dataset.py`
- [x] Implementar `src/quality/model.py` (MobileNetV2 transfer learning)
- [x] Implementar `src/quality/train.py`
- [x] Notebook `04_quality_model.ipynb` — treino e avaliação (ROC, confusion matrix, training curves)

### Parsers

- [x] Implementar `src/parsing/schema.py` (modelos Pydantic: Receipt, LineItem)
- [x] Implementar `src/parsing/regex_parser.py`
- [x] Implementar `src/parsing/llm_parser.py` (Gemini 3.5 Flash, temperature=0, JSON mode)
- [x] Testar ambos parsers em subset com ground truth

---

## 🔗 Semana 3: Integração + Experimentos

### Pipeline

- [x] Implementar `src/pipeline.py` (orquestração completa: img → JSON)
- [x] Implementar `src/evaluation/metrics.py` (CER, WER, F1, ANLS, Exact Match)
- [x] Implementar `src/evaluation/visualization.py` (gráficos)

### Experimentos (13 total)

- [x] Exp 1: Sem pré-processamento vs com
- [x] Exp 2: CLAHE vs sem CLAHE
- [x] Exp 3: Adaptive Threshold vs sem
- [x] Exp 4: Deskew on vs off
- [x] Exp 5: Ablation — pipeline completa vs subconjuntos
- [x] Exp 6: PaddleOCR vs Tesseract (CER, WER, latência)
- [x] Exp 7: Pipeline com CNN filter vs sem
- [x] Exp 8: Threshold de rejeição CNN (0.3 / 0.5 / 0.7)
- [x] Exp 9: Regex vs LLM (F1, latência, custo)
- [x] Exp 10: LLM zero-shot vs few-shot (3 exemplos)
- [x] Exp 11: OCR ground truth + Parser vs OCR real + Parser
- [x] Exp 12: Pipeline completa vs baseline naive
- [x] Exp 13: Diferentes resoluções (300 / 600 / original DPI)

### Notebooks

- [x] Notebook `03_ocr_comparison.ipynb`
- [x] Notebook `05_parser_comparison.ipynb`

---

## 📊 Semana 4: Finalização

### Resultados e Visualizações

- [x] Notebook `06_final_results.ipynb` (consolidação)
- [x] Gerar bar chart: CER por técnica de pré-processamento
- [x] Gerar heatmap: F1 por campo × abordagem
- [x] Gerar box plot: distribuição de scores de qualidade
- [x] Gerar ROC curve da CNN
- [x] Gerar confusion matrix da CNN
- [x] Gerar line chart: F1 vs threshold de rejeição
- [x] Gerar scatter plot: CNN score vs CER
- [x] Gerar radar chart: Regex vs LLM
- [x] Gerar waterfall chart: contribuição de cada etapa no F1
- [x] Gerar grid de imagens: antes/depois pré-processamento
- [x] Gerar training curves: loss/accuracy da CNN

### Imagens para Relatório

- [x] Pipeline visual de 1 imagem (boa, média, ruim) por cada etapa
- [x] Histograma antes/depois de CLAHE
- [x] Deskew antes/depois
- [x] Bounding boxes do OCR sobrepostos
- [x] Grad-CAM da CNN
- [x] Exemplos de falha (OCR errado, parser errado)
- [x] Diagrama da arquitetura da CNN

### Documentação e Polish

- [x] Escrever relatório final
- [ ] Polir README (badges, GIF demo, screenshots, tabela de resultados)
- [ ] Testes unitários (`tests/test_preprocessing.py`, `test_ocr.py`, `test_parsing.py`)
- [ ] Diagrama de arquitetura no README (mermaid)

### Stretch Goals

- [ ] FastAPI demo endpoint ou Streamlit app
- [ ] GitHub Actions (lint ruff + pytest)
- [ ] Docker para reprodutibilidade
- [ ] Colab badge nos notebooks
- [ ] Release v1.0 com modelo treinado como asset

---

## 📌 Prioridade se faltar tempo

1. ✅ Pré-processamento + OCR + Regex parser (MVP)
2. ✅ LLM parser + comparação
3. ✅ CNN de qualidade
4. ✅ Polish (README, testes, docs)
