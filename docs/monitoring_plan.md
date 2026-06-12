# Plano de Monitoramento — Recomendador Neural E-commerce

---

## 1. Métricas Online (produção)

| Métrica | Alvo | Alerta |
|---|---|---|
| CTR (Click-Through Rate) das recomendações | > 8% | < 5% |
| Conversion Rate itens recomendados | > 2% | < 1% |
| Latência de inferência (p99) | < 100ms | > 200ms |
| Taxa de cold-start (usuários sem histórico) | < 20% | > 35% |

---

## 2. Métricas Offline (re-avaliação periódica)

Recalcular semanalmente no test set com novos dados:

| Métrica | Alvo inicial | Degradação aceitável |
|---|---|---|
| NDCG@10 | Valor do treino inicial | -5% |
| Precision@10 | Valor do treino inicial | -5% |
| Hit Rate@10 | Valor do treino inicial | -10% |

---

## 3. Data Drift

- Monitorar distribuição de `user_id` e `item_id` no tráfego real vs. dados de treino
- Alertar quando > 30% de requisições são de usuários sem histórico no modelo atual
- Monitorar surgimento de novos itens não vistos no treino (catalog drift)

---

## 4. Retreinamento

| Gatilho | Ação |
|---|---|
| NDCG@10 cai > 5% por 3 dias consecutivos | Acionar retreinamento |
| > 20% de novos itens no catálogo | Retreinamento com dados incrementais |
| 30 dias sem retreinamento | Retreinamento preventivo |

### Fluxo de promoção no MLflow Model Registry
1. Novo modelo treinado → versão em `None` (candidato)
2. Avaliação offline aprovada → promover para `Staging`
3. Teste A/B em 10% do tráfego → promover para `Production`
4. Modelo antigo → rebaixar para `Archived`

---

## 5. Ferramentas

- **MLflow**: tracking de experimentos e Model Registry
- **DVC**: versionamento de dados e reprodutibilidade do pipeline
- **Evidently** (recomendado para futuro): relatórios de data drift automatizados
