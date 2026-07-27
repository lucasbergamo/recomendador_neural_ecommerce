# Model Card — Neural Collaborative Filtering (NCF)

> Formato baseado em Model Cards for Model Reporting (Mitchell et al., 2019, Google).

---

## 1. Detalhes do Modelo

| Atributo | Valor |
|---|---|
| Nome | Neural Collaborative Filtering (NeuMF) |
| Versão | 1.1.0 — pós-tuning (21/07/2026) |
| Tipo | Rede Neural — Recomendação Implícita (ranking) |
| Framework | PyTorch 2.x |
| Arquivo | `models/ncf.pt` |
| Registry | `ncf-recommender`, aliases `@staging`/`@production` (MLflow Model Registry) |
| Desenvolvido por | Projeto Tech Challenge Fase 02 — FIAP Pós-Tech MLET |
| Data | Junho–Julho 2026 |

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
| embedding_dim | **32** (era 64) | Reduzido após tuning — dataset pequeno/esparso (943 usuários, 1.349 itens pós-filtro) não sustenta 64 dims sem overfitting; ver seção 4.1 |
| mlp_layers | [128, 64, 32] | Funil progressivo — padrão NCF original (He et al.) |
| dropout | 0.3 | Regularização para dataset de interações esparsas |
| optimizer | Adam, lr=0.001 | Padrão para modelos baseados em embeddings — testado lr=0.0003 no tuning, não venceu (seção 4.1) |
| loss | BCEWithLogitsLoss | Feedback implícito binário (interagiu / não interagiu) |
| batch_size | 256 | Adequado para datasets de 100k interações |
| early_stopping | patience=10 | Evita overfitting em datasets de usuário-item |
| negative_ratio | 4 | Testado ratio=8 no tuning, não venceu (seção 4.1) |

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

| Atributo | Bruto (MovieLens 100K oficial) | Pós-filtro (usado no treino) |
|---|---|---|
| Interações | 100.000 ratings | 99.287 linhas |
| Usuários | 943 | 943 (sem filtro de cold-start) |
| Itens | 1.682 filmes | 1.349 (333 removidos — cold-start, <5 avaliações) |

| Split | Composição |
|---|---|
| Treino | ~80% das interações positivas por usuário (mais antigas) + negativos amostrados (1:4) |
| Validação | 10% (temporal, por usuário) — só positivos |
| Teste | 10% mais recentes por usuário (temporal) — só positivos |

Split **leave-time-out por usuário**: evita vazamento de dados (o modelo nunca vê o "futuro" de um usuário durante o treino) e simula o cenário real de produção.

---

## 4. Resultados de Avaliação

Métricas de ranking (`@10`) calculadas via leave-one-out por usuário no conjunto de teste, exceto onde indicado (`val_*`).

| Modelo | NDCG@10 | Precision@10 | Recall@10 | HR@10 | MAP@10 |
|---|---|---|---|---|---|
| **NCF (NeuMF)** | **0.0345** | 0.0231 | 0.0445 | 0.1898 | 0.0148 |
| Popularity | **0.0350** | 0.0244 | 0.0396 | 0.1983 | 0.0153 |
| SVD | 0.0079 | 0.0051 | 0.0132 | 0.0509 | 0.0031 |
| Random | 0.0061 | 0.0052 | 0.0064 | 0.0509 | 0.0020 |

**Leitura honesta:** o NCF fica **marginalmente abaixo do Popularity** em 4 das 5 métricas (só ganha em Recall@10), mesmo depois do tuning. Isso é esperado e documentado como trade-off — ver seção 5.

### 4.1 Tuning — processo e evidência

Ponto de partida (config original, pré-tuning): `embedding_dim=64`, `lr=0.001`, `negative_ratio=4` → `test_ndcg_at_10=0.0328`.

Três configs testadas isolando uma variável cada, ~3min de treino cada, evidência completa nos runs do MLflow (experimento `recommendation-system`):

| Config | val_ndcg_at_10 | test_ndcg_at_10 | Observação |
|---|---|---|---|
| 1 — `negative_ratio=8` | **0.0370** (melhor por val) | 0.0323 (pior das 3 por test) | Divergência val/test — sinal de instabilidade do split de validação (~9.531 positivos), não adotada apesar de vencer por val |
| **2 — `embedding_dim=32`** | 0.0353 | **0.0345** | **Escolhida** — única com val e test consistentes entre si, e melhor que a config original nos dois |
| 3 — `learning_rate=0.0003` | 0.0364 | 0.0334 | Intermediária, val/test também divergem, mas menos que a config 1 |

**Critério de seleção:** por `val_ndcg_at_10` (não test — evita escolher hiperparâmetro otimizando pra um conjunto que devia ficar "não visto"). A config 1 venceria por esse critério puro, mas seu test caiu abaixo até da config original — por isso o critério final combinou "melhor val" **com** "consistência val/test", não val isolado. O test das 3 foi consultado nessa análise (não é seleção cega por val); a ressalva fica registrada aqui explicitamente, como listado nas limitações.

**Resultado:** nenhuma das 3 configs bateu Popularity (`ndcg_at_10=0.0350`), mas a config 2 trouxe o NCF de `0.0328` pra `0.0345` — **~5% de melhora relativa**, reduzindo a distância pro baseline sem eliminá-la. Mantido o trade-off, agora com evidência experimental em vez de uma única tentativa.

---

## 5. Limitações Conhecidas

- **NCF perde de Popularity em ranking** (NDCG/Precision/HR/MAP) mesmo após tuning — dataset pequeno e denso favorece recomendação não-personalizada (poucos itens concentram a maioria das interações); ganho do NCF viria de mais dados ou features de conteúdo, fora do escopo desta entrega
- **Seleção de hiperparâmetro consultou o test set na análise final** (config 2 foi escolhida também por ter test consistente com val, não só por val) — desvio pequeno mas real do protocolo estritamente "val-only"; documentado em vez de omitido
- **Split de validação instável para alguns hiperparâmetros** — evidenciado pela config 1 (negative_ratio=8), que teve o melhor val e o pior test das 3; ~9.531 positivos de validação parecem insuficientes para estimar generalização de forma robusta em todas as configs
- Problema de cold-start: usuários e itens novos sem histórico não podem ser recomendados
- Feedback implícito: "não interagiu" ≠ "não gosta" — negativos são amostrados, não observados
- Dataset de filmes (MovieLens) adaptado para o contexto de e-commerce do desafio: distribuição de interações pode diferir de produtos reais
- Sem features de conteúdo: a arquitetura NCF usa apenas IDs, não descrição/categoria de produtos
- Não-determinismo residual entre ambientes (CPU/threading) — variações de ±0.001–0.002 em métricas são esperadas entre reexecuções, mesmo com seed fixa

---

## 6. Considerações Éticas

- O sistema pode criar bolhas de recomendação (filter bubbles) ao reforçar padrões existentes
- Grupos com menos histórico de interações (usuários novos) recebem recomendações piores
- Recomendação: auditar periodicamente a diversidade das recomendações por segmento

---

## 7. Como Reproduzir

```bash
git clone https://github.com/lucasbergamo/recomendador_neural_ecommerce.git
cd recomendador_neural_ecommerce
poetry install
cp .env.example .env

make docker-up              # sobe o MLflow (localhost:5000)
dvc repro                   # pipeline completo: preprocess → feature_eng →
                             # {train_baselines, train} → evaluate
make register                # registra o modelo final no Model Registry

mlflow ui                   # se preferir MLflow local em vez do container
```

O remote do DVC é local (`~/dvc-storage`, simula um bucket S3) — quem clona o
repo não tem acesso a ele, então `dvc pull` não vai funcionar. Isso é
intencional: `dvc repro` reconstrói tudo do zero a partir dos CSVs brutos do
MovieLens já commitados em `data/bronze/`, sem depender de nenhum remote
externo. Ver `README.md` para o passo a passo completo.
