# Dataset — MovieLens 100K

> Documentação separada do README conforme boa prática de organização de projetos ML.

---

## Fonte e Acesso

| Atributo | Detalhe |
|---|---|
| Nome | MovieLens 100K |
| Organização | GroupLens Research Lab — University of Minnesota |
| URL | https://grouplens.org/datasets/movielens/100k/ |
| Licença | Uso acadêmico e pesquisa — sem redistribuição comercial |
| Download automático | `make data` (via `src/data/load.py`) |

---

## Estatísticas Básicas

| Métrica | Valor |
|---|---|
| Total de ratings | 100.836 |
| Usuários únicos | 943 |
| Itens (filmes) únicos | 1.682 |
| Densidade da matriz | 6,3% (esparsa) |
| Escala de ratings | 1 a 5 (inteiros) |
| Período | 01/09/1997 a 22/04/1998 |

---

## Estrutura dos Arquivos

### `data/bronze/ratings.csv` (gerado do u.data)
| Coluna | Tipo | Descrição |
|---|---|---|
| user_id | int | ID do usuário (1–943) |
| item_id | int | ID do filme (1–1682) |
| rating | float | Avaliação de 1 a 5 |
| timestamp | datetime | Data/hora da avaliação |

### `data/bronze/items.csv` (gerado do u.item)
| Coluna | Tipo | Descrição |
|---|---|---|
| item_id | int | ID do filme |
| title | str | Título com ano (ex: "Toy Story (1995)") |
| release_date | str | Data de lançamento |
| action, comedy, drama, ... | int | Flags binárias de gênero |

### `data/bronze/users.csv` (gerado do u.user)
| Coluna | Tipo | Descrição |
|---|---|---|
| user_id | int | ID do usuário |
| age | int | Idade |
| gender | int | 0=Masculino, 1=Feminino |
| occupation | str | Ocupação |
| zip_code | str | CEP (não utilizado no modelo) |

---

## Pipeline de Pré-processamento

### Bronze → Silver (`src/data/preprocess.py`)
1. Remove registros com `user_id`, `item_id` ou `rating` nulos
2. Filtra usuários e itens com < 5 interações (cold-start filter)
3. Re-indexa IDs para índices contíguos começando em 0 (requisito para embeddings PyTorch)
4. Converte `timestamp` para `datetime64`
5. Codifica `gender` como binário (M=0, F=1)

### Silver → Gold (`src/data/features.py`)
1. Converte ratings explícitos → feedback implícito binário (interaction=1)
2. Adiciona amostras negativas: 4 itens não visitados por item positivo
3. Split temporal: últimas 10% interações de cada usuário → test, 10% anteriores → val
4. Garante que todos os usuários estão representados em treino

---

## Considerações sobre Representatividade

- Dataset coletado em 1997–1998: preferências de filmes mudaram significativamente
- 69% dos usuários são do gênero masculino — viés de gênero nas recomendações
- Filmes populares concentram a maioria das avaliações (distribuição power-law)
- Sem dados demográficos de renda ou localização geográfica

---

## Alternativas de Dataset para E-commerce Real

| Dataset | Tamanho | Contexto | Acesso |
|---|---|---|---|
| RetailRocket | 2.8M eventos | E-commerce RU | Kaggle |
| Instacart Market Basket | 3.4M pedidos | Supermercado | Kaggle |
| Amazon Product Reviews | >180M | E-commerce US | HuggingFace |
