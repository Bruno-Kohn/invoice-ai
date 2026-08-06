# Invoice AI — Project Plan

> Pipeline completa de Document AI para extração estruturada de notas fiscais: imagem → pré-processamento (OpenCV) → avaliação de qualidade (CNN) → OCR (PaddleOCR) → parsing (Regex vs LLM) → JSON → métricas. Projeto estruturado como software profissional com experimentos comparativos rigorosos.

---

## Índice

1. [Crítica da Ideia Proposta](#1-crítica-da-ideia-proposta)
2. [Técnicas de Pré-processamento](#2-técnicas-de-pré-processamento--análise-crítica)
3. [Análise de OCR Engines](#3-análise-de-ocr-engines)
4. [CNN de Qualidade](#4-cnn-de-qualidade--análise-de-alternativas)
5. [Comparação Regex vs LLM](#5-comparação-regex-vs-llm--metodologia-científica)
6. [Arquitetura Final](#6-arquitetura-final)
7. [Estrutura de Pastas](#7-estrutura-de-pastas)
8. [Schema JSON (Pydantic)](#8-schema-json-pydantic)
9. [Métricas Científicas](#9-métricas-científicas)
10. [Experimentos](#10-experimentos)
11. [Gráficos para o Relatório](#11-gráficos-para-o-relatório)
12. [Imagens a Salvar para Relatório](#12-imagens-a-salvar-para-relatório)
13. [Cronograma (4 semanas)](#13-cronograma-4-semanas)
14. [Riscos Técnicos e Mitigações](#14-riscos-técnicos-e-mitigações)
15. [Portfólio GitHub](#15-portfólio-github--diferenciadores)
16. [Extensões Futuras](#16-extensões-futuras-produto-real)
17. [Decisões Consolidadas](#17-decisões-consolidadas)

---

## 1. Crítica da Ideia Proposta

### Pontos fortes

- Pipeline end-to-end demonstra domínio completo do fluxo Document AI.
- Comparação Regex vs LLM é relevante e atual na indústria.
- CNN para qualidade é viável em 4 semanas e demonstra capacidade de treinar modelos próprios.
- Stack tecnológica coerente e moderna.

### Pontos a melhorar

- **Dataset**: SROIE tem apenas ~626 imagens (receipts de Singapura, inglês). Recomendo **CORD-v2** como dataset principal — 1000 imagens com anotações ricas (30 campos estruturados), licença CC-BY-4.0, amplamente usado em benchmarks.
- **CNN de qualidade**: falta dataset anotado. Solução: gerar degradações sintéticas em imagens boas do CORD (blur, rotação, noise, compressão JPEG) — cria dataset automaticamente.
- **Falta pós-processamento de OCR**: correção de erros comuns (O→0, l→1, espaços indevidos) é essencial entre OCR e parser.
- **Falta Document Detection**: antes de qualquer processamento, isolar o documento do fundo (contour detection + perspective transform).
- **Escopo de pré-processamento excessivo**: nem todas as técnicas fazem sentido. Detalhado na seção 2.

---

## 2. Técnicas de Pré-processamento — Análise Crítica

### Essenciais para documentos

| Técnica | Justificativa |
|---------|---------------|
| Grayscale | OCR opera em 1 canal; reduz complexidade |
| CLAHE | Superior a histogram equalization para iluminação irregular |
| Gaussian Blur leve (3×3) | Remove ruído antes de threshold |
| Adaptive Threshold | Superior a Otsu para iluminação não-uniforme |
| Morphological Closing | Conecta caracteres quebrados |
| Deskew (Hough Transform / minAreaRect) | Rotação >2° degrada OCR significativamente |
| Perspective Correction | Essencial para fotos de celular |

### Úteis em cenários específicos

| Técnica | Quando |
|---------|--------|
| Bilateral Filter | Preserva bordas em fotos de baixa qualidade |
| Median Blur | Ruído salt-and-pepper |
| Canny + Contour Detection | Isolar documento do fundo |

### Desnecessárias / Prejudiciais

| Técnica | Motivo |
|---------|--------|
| Histogram Equalization global | CLAHE é estritamente superior |
| Otsu isolado | Adaptive threshold superior em cenários reais |
| Erosion/Dilation isoladas | Destroem ou fundem caracteres |

---

## 3. Análise de OCR Engines

| Engine | Prós | Contras | Veredito |
|--------|------|---------|----------|
| **PaddleOCR** | SOTA em benchmarks, detection+recognition integrado, 80+ idiomas, leve | Docs parcialmente em chinês | **✓ Escolha principal** |
| Tesseract | Maduro, amplamente usado, fácil de instalar | Requer binarização perfeita, sem detection, inferior em fotos | **Baseline para comparação** |
| EasyOCR | API simples | Mais lento, menos preciso | ❌ Não recomendado |
| TrOCR | Transformer-based | Apenas recognition, pesado | ❌ Fora do escopo |
| Donut | End-to-end sem OCR | Requer fine-tuning pesado, GPU intensivo | ⚠️ Stretch goal |

---

## 4. CNN de Qualidade — Análise de Alternativas

| Alternativa | Viável? | Justificativa |
|-------------|---------|---------------|
| **OCR-Readiness score (binário/regressão)** | ✅ **Recomendado** | Dataset gerável sinteticamente, mensurável (correlacionar score com CER), útil na pipeline real |
| Detectar blur | ❌ | Laplacian variance resolve sem NN — trivial |
| Classificar tipo de documento | ❌ | Irrelevante se usar só receipts |
| Detectar rotação | ❌ | Solucionável com Hough sem NN |
| Layout segmentation | ⚠️ | Impressionante mas complexo demais para 4 semanas |

### Arquitetura recomendada

MobileNetV2 (transfer learning ImageNet) → fine-tune com ~5000 imagens sintéticas (1000 originais × 5 degradações).

### Degradações para gerar dataset

- Gaussian blur (σ=1, 3, 5, 7) e motion blur
- Rotação (5°, 10°, 15°, 30°)
- Ruído gaussiano e salt-and-pepper
- Downscale + upscale (simular baixa resolução)
- Compressão JPEG (quality=10, 20, 30)
- Iluminação irregular (gradientes)
- Oclusão parcial

### Labeling automático

Rodar OCR na imagem degradada, calcular CER vs ground truth:

- CER < 5% → `ready`
- CER > 20% → `not_ready`
- Entre → `marginal`

---

## 5. Comparação Regex vs LLM — Metodologia Científica

### Design experimental

1. **Ground truth**: Anotações do CORD-v2.
2. **Mesma entrada**: Ambos parsers recebem o mesmo output de OCR (texto + bounding boxes).
3. **Determinismo**: LLM com temperature=0, seed fixo.
4. **Estratificação**: Medir por qualidade da imagem (boa/média/ruim) e por complexidade (poucos vs muitos items).
5. **Múltiplas execuções**: 3 runs para medir variância.

### LLM recomendada

GPT-4o-mini via API (~$0.15/1M input tokens) — custo baixo, JSON mode nativo, determinístico.

### Métricas de comparação

| Métrica | Descrição |
|---------|-----------|
| Field-level F1 | Por campo individual |
| ANLS (Average Normalized Levenshtein Similarity) | Métrica padrão SROIE/CORD |
| Exact Match | % de JSONs perfeitos |
| Latência (ms/doc) | Tempo de parsing |
| Custo ($/doc) | Para LLM |
| Robustez | F1 quando CER do OCR > 10% |

---

## 6. Arquitetura Final

```
Input Image
     │
     ▼
┌─────────────────────┐
│ Document Detection    │  ← Contour + perspective transform
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Quality Assessment    │  ← CNN MobileNetV2 → score 0-1
└──────────┬──────────┘
           │ score < threshold → REJECT
           ▼
┌─────────────────────┐
│ Pre-processing        │  ← CLAHE, deskew, denoise (configurável)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ OCR                   │  ← PaddleOCR (+ Tesseract para comparação)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ OCR Post-processing   │  ← Correção de caracteres, merge de boxes
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│  Regex   │ │   LLM   │
│  Parser  │ │  Parser │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────────────────┐
│ Validation (Pydantic) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Evaluation & Metrics  │
└───────────────────────┘
```

---

## 7. Estrutura de Pastas

```
invoice-ai/
├── configs/
│   ├── pipeline.yaml
│   ├── preprocessing.yaml
│   └── experiments.yaml
├── data/
│   ├── raw/                   # CORD-v2 original
│   ├── processed/
│   ├── synthetic/             # Dataset para CNN
│   └── annotations/
├── docs/
│   ├── project-plan.md
│   ├── architecture.md
│   └── decisions.md
├── models/
│   └── quality_cnn/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ocr_comparison.ipynb
│   ├── 04_quality_model.ipynb
│   ├── 05_parser_comparison.ipynb
│   └── 06_final_results.ipynb
├── results/
│   ├── metrics/
│   ├── figures/
│   └── samples/
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── document_detector.py
│   │   ├── deskew.py
│   │   ├── enhancement.py
│   │   └── transforms.py
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── dataset.py
│   │   └── train.py
│   ├── ocr/
│   │   ├── __init__.py
│   │   ├── engine.py          # Interface abstrata
│   │   ├── paddle_ocr.py
│   │   ├── tesseract_ocr.py
│   │   └── postprocessing.py
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── regex_parser.py
│   │   ├── llm_parser.py
│   │   └── schema.py         # Pydantic models
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── config.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_ocr.py
│   └── test_parsing.py
├── .gitignore
├── README.md
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## 8. Schema JSON (Pydantic)

```python
class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total_price: float

class Receipt(BaseModel):
    company_name: Optional[str] = None
    company_id: Optional[str] = None       # CNPJ / tax ID
    address: Optional[str] = None
    date: Optional[str] = None             # ISO format
    time: Optional[str] = None
    receipt_number: Optional[str] = None
    items: list[LineItem] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: float
    payment_method: Optional[str] = None
    change: Optional[float] = None
    currency: Optional[str] = None
```

---

## 9. Métricas Científicas

### OCR-level

| Métrica | Descrição | Referência |
|---------|-----------|------------|
| CER (Character Error Rate) | Edit distance / comprimento GT | Padrão em papers de OCR |
| WER (Word Error Rate) | Word-level edit distance | Padrão em papers de OCR |
| Detection IoU | Precision/recall dos bounding boxes | ICDAR benchmarks |

### Parser-level

| Métrica | Descrição |
|---------|-----------|
| Field-level F1 | Por campo individual |
| Macro F1 | Média dos F1 de todos os campos |
| ANLS (Average Normalized Levenshtein Similarity) | Métrica padrão ICDAR/SROIE |
| Exact Match (EM) | % de documentos com extração perfeita |
| TED (Tree Edit Distance) | Para lista de items |

### Pipeline-level

| Métrica | Descrição |
|---------|-----------|
| End-to-end F1 | Desde imagem até JSON final |
| Latência (ms/doc) | Tempo total de processamento |
| Throughput (docs/min) | Capacidade de processamento |
| Rejection rate | % de imagens rejeitadas pela CNN |

---

## 10. Experimentos

| # | Experimento | O que mede |
|---|-------------|------------|
| 1 | Sem pré-processamento vs com | Impacto no CER |
| 2 | CLAHE vs sem CLAHE | Impacto em imagens escuras |
| 3 | Adaptive Threshold vs sem | Impacto geral |
| 4 | Deskew on vs off | Impacto em imagens rotacionadas |
| 5 | Ablation: pipeline completa vs subconjuntos | Contribuição de cada etapa |
| 6 | PaddleOCR vs Tesseract | CER, WER, latência |
| 7 | Pipeline com CNN filter vs sem | Qualidade do output final |
| 8 | Threshold de rejeição CNN (0.3 / 0.5 / 0.7) | Trade-off rejection vs quality |
| 9 | Regex vs LLM | F1, latência, custo |
| 10 | LLM zero-shot vs few-shot (3 exemplos) | Impacto de exemplos |
| 11 | OCR ground truth + Parser vs OCR real + Parser | Isolamento de fonte de erro |
| 12 | Pipeline completa vs baseline naive | Valor agregado total |
| 13 | Diferentes resoluções de input (300 / 600 / original DPI) | Impacto de resolução |

---

## 11. Gráficos para o Relatório

1. **Bar chart**: CER por técnica de pré-processamento
2. **Heatmap**: F1 por campo × abordagem (regex vs LLM)
3. **Box plot**: Distribuição de scores de qualidade
4. **ROC curve**: CNN de qualidade
5. **Confusion matrix**: CNN
6. **Line chart**: F1 vs threshold de rejeição
7. **Scatter plot**: CNN score vs CER
8. **Radar chart**: Regex vs LLM (F1, latência, custo, robustez)
9. **Waterfall chart**: Contribuição de cada etapa no F1
10. **Grid de imagens**: Antes/depois de pré-processamento
11. **Training curves**: Loss/accuracy da CNN

---

## 12. Imagens a Salvar para Relatório

- Pipeline visual de 1 imagem (boa, média, ruim) passando por cada etapa
- Histograma antes/depois de CLAHE
- Deskew antes/depois
- Bounding boxes do OCR sobrepostos
- Grad-CAM da CNN (o que a rede "olha")
- Exemplos de falha (OCR errado, parser errado, extração parcial)
- Diagrama da arquitetura da CNN

---

## 13. Cronograma (4 semanas)

### Semana 1: Fundação

- [ ] Setup do projeto (estrutura, git, README, requirements)
- [ ] Download e EDA do CORD-v2 (notebook 01)
- [ ] Implementar módulo de pré-processamento completo
- [ ] Implementar wrappers OCR (PaddleOCR + Tesseract)
- [ ] Primeiros experimentos de pré-processamento (notebook 02)

### Semana 2: CNN + Parsers

- [ ] Gerar dataset sintético (degradações)
- [ ] Treinar CNN MobileNetV2 (notebook 04)
- [ ] Implementar Regex parser
- [ ] Implementar LLM parser
- [ ] Schema Pydantic + validação

### Semana 3: Integração + Experimentos

- [ ] Integrar pipeline completa (`pipeline.py`)
- [ ] Rodar todos os 13 experimentos
- [ ] Coletar métricas e gerar visualizações
- [ ] Notebook 03 (OCR) e 05 (parsers)

### Semana 4: Finalização

- [ ] Notebook 06 (resultados consolidados)
- [ ] Escrever relatório
- [ ] Polir README (badges, GIF demo, screenshots)
- [ ] Testes unitários
- [ ] (Stretch) FastAPI demo endpoint ou Streamlit app

---

## 14. Riscos Técnicos e Mitigações

| Risco | Prob. | Impacto | Mitigação |
|-------|-------|---------|-----------|
| Anotações CORD-v2 incompatíveis com schema | Média | Alto | Verificar formato no dia 1 da EDA |
| PaddleOCR difícil de instalar (deps conflitantes) | Média | Médio | Ter EasyOCR como fallback; Docker |
| CNN não converge | Baixa | Alto | Transfer learning + dataset grande; fallback: Laplacian variance heurística |
| Custo da API LLM | Baixa | Médio | GPT-4o-mini (~$0.15/1M tokens); limitar a 100 docs |
| Tempo insuficiente | Média | Alto | Priorizar pipeline funcional sem CNN primeiro; CNN é extensão |

### Ordem de prioridade se faltar tempo

1. Pré-processamento + OCR + Regex parser (MVP)
2. LLM parser + comparação
3. CNN de qualidade
4. Polish (README, testes, docs)

---

## 15. Portfólio GitHub — Diferenciadores

1. **README com GIF demo** mostrando a pipeline em ação
2. **Badges**: Python version, license, CI status
3. **GitHub Actions**: lint (ruff) + tests (pytest) em cada push
4. **Resultados no README**: Tabela resumida de F1 por abordagem
5. **Conventional Commits**: histórico semântico
6. **Makefile**: `make train`, `make evaluate`, `make demo`
7. **Docker**: reprodutibilidade garantida
8. **Release v1.0**: modelo treinado como asset
9. **Diagrama de arquitetura** no README (mermaid ou imagem)
10. **Colab badge**: notebook executável na nuvem

---

## 16. Extensões Futuras (Produto Real)

1. FastAPI + Streamlit frontend (upload → JSON)
2. Suporte a notas fiscais brasileiras (CNPJ, CPF, layout NF-e)
3. Fine-tuning PaddleOCR para domínio específico
4. Active Learning (usuário corrige → retrain)
5. LayoutLMv3 / Donut como abordagem end-to-end
6. Batch processing com Celery/Redis
7. Mobile SDK com feedback de qualidade em tempo real
8. Multi-language support

---

## 17. Decisões Consolidadas

| Decisão | Justificativa |
|---------|---------------|
| Dataset: CORD-v2 | 1000 imgs, 30 campos, CC-BY-4.0, benchmark estabelecido |
| OCR: PaddleOCR + Tesseract baseline | SOTA vs estabelecido para comparação |
| CNN: MobileNetV2 transfer learning | Leve, eficiente, convergência rápida |
| LLM: GPT-4o-mini | Barato, JSON mode, determinístico |
| Framework: PyTorch (torchvision) | Ecossistema transfer learning |
| Validação: Pydantic v2 | Type safety + serialização |
| Config: YAML | Legível, versionável |
