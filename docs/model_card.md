# Model Card — Neural Collaborative Filtering (NCF)

> Formato baseado em Model Cards for Model Reporting (Mitchell et al., 2019, Google).

---

## 1. Detalhes do Modelo

| Atributo | Valor |
|---|---|
| Nome | Neural Collaborative Filtering (NeuMF) |
| Versão | 1.0.0 |
| Tipo | Rede Neural — Recomendação Implícita (ranking) |
| Framework | PyTorch 2.x |
| Arquivo | `models/ncf.pt` |
| Desenvolvido por | Projeto Tech Challenge Fase 02 — FIAP Pós-Tech MLET |
| Data | Junho 2026 |

### Arquitetura

```
User ID → Embedding (GMF) ─┐
Item ID → Embedding (GMF) ─┤─ Element-wise product (GMF path) ──┐
                            │                                     ├─ Linear(1) → logit
User ID → Embedding (MLP) ─┤─ Concat → Linear(128) → ReLU → ... ┘
Item ID → Embedding (MLP) ─┘   → Linear(64) → Linear(32)
```

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| embedding_dim | 64 | Balanço entre capacidade representacional e overfitting |
| mlp_layers | [128, 64, 32] | Funil progressivo — padrão NCF original (He et al.) |
| dropout | 0.3 | Regularização para dataset de interações esparsas |
| optimizer | Adam, lr=0.001 | Padrão para modelos baseados em embeddings |
| loss | BCEWithLogitsLoss | Feedback implícito binário (interagiu / não interagiu) |
| batch_size | 256 | Adequado para datasets de 100k interações |
| early_stopping | patience=10 | Evita overfitting em datasets de usuário-item |

---

## 2. Uso Pretendido

### Uso adequado
- Recomendar produtos relevantes na página inicial ou durante navegação
- Personalizar resultados de busca por perfil de usuário
- Apoiar estratégias de cross-selling e upselling

### Uso inadequado
- ❌ Tomar decisões discriminatórias baseadas em histórico de navegação
- ❌ Inferir informações sensíveis (renda, localização) a partir do padrão de compras
- ❌ Aplicar em domínios radicalmente diferentes do treino sem retreinamento

---

## 3. Dados de Treinamento

> Documentação detalhada do dataset disponível em `docs/dataset.md`

| Atributo | Detalhe |
|---|---|
| Dataset | MovieLens 100K |
| Total de interações | 100.836 ratings |
| Usuários | 943 |
| Itens | 1.682 filmes |
| Split treino | 80% (temporal — interações mais antigas) |
| Split validação | 10% (temporal) |
| Split teste | 10% (interações mais recentes por usuário) |
| Balanceamento | Feedback implícito: 1 positivo : 4 negativos amostrados |

---

## 4. Resultados de Avaliação

*A ser preenchido após o treinamento (Etapa 4).*

| Modelo | NDCG@10 | Precision@10 | Recall@10 | HR@10 | MAP@10 |
|---|---|---|---|---|---|
| **NCF (NeuMF)** | — | — | — | — | — |
| SVD | — | — | — | — | — |
| Popularity | — | — | — | — | — |
| Random | — | — | — | — | — |

---

## 5. Limitações Conhecidas

- Problema de cold-start: usuários e itens novos sem histórico não podem ser recomendados
- Feedback implícito: "não clicou" ≠ "não gosta" — negativos são amostrados, não observados
- Dataset de filmes adaptado para e-commerce: distribuição de interações pode diferir de produtos reais
- Sem features de conteúdo: a arquitetura NCF usa apenas IDs, não descrição de produtos

---

## 6. Considerações Éticas

- O sistema pode criar bolhas de recomendação (filter bubbles) ao reforçar padrões existentes
- Grupos com menos histórico de interações (usuários novos) recebem recomendações piores
- Recomendação: auditar periodicamente a diversidade das recomendações por segmento

---

## 7. Como Reproduzir

```bash
git clone https://github.com/lucascbergamo/recomendador_neural_ecommerce.git
cd recomendador_neural_ecommerce
poetry install
cp .env.example .env
make data          # download + pipeline completo
make train         # treina NCF
make eval          # avalia no test set
mlflow ui          # visualiza experimentos em localhost:5000
```
