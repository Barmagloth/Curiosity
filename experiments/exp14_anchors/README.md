# Exp14: Anchors + Periodic Rebuild

**Status:** CONDITIONAL PASS (720 configs, grid PASS, graph/tree FAIL)
**Roadmap level:** P1-B3 (depends on exp11 dirty signatures + exp13 segment compression)

---

# RU

## Проблема

При инкрементальном уточнении дерево обновляется локально (local update) без полного перестроения. Вопрос: насколько расходятся результаты local update и full rebuild? Какая стратегия rebuild минимизирует divergence при минимальных затратах?

## Метод

720 конфигураций: 4 пространства x 9 стратегий x 20 seeds. 50 шагов на конфиг.

**Стратегии rebuild:**
- `no_rebuild` — только local update
- `periodic_{5,10,20,50}` — полный rebuild каждые K шагов
- `dirty_{0.05,0.1,0.2,0.5}` — rebuild при >X% dirty-узлов

## Kill Criteria и результаты

**Kill criterion:** divergence < 5% (0.05).

| Пространство | Max Div | Mean Div | Лучшая стратегия | Вердикт |
|---|---|---|---|---|
| scalar_grid | 0.000 | 0.000 | no_rebuild | PASS |
| vector_grid | 0.000 | 0.000 | no_rebuild | PASS |
| irregular_graph | 1.517 | 0.374 | dirty_0.05 (0.204 mean) | FAIL |
| tree_hierarchy | 1.715 | 0.694 | dirty_0.05 (0.204 mean) | FAIL |

## Ключевые находки

- **Сетки (scalar_grid, vector_grid):** divergence = 0.000 для ВСЕХ стратегий. Local update идеален — rebuild не нужен.
- **Graph и tree:** dirty-triggered (порог 0.05) — лучшая стратегия (mean_div=0.204), но далеко от порога 0.05. Periodic стратегии хуже dirty-triggered.
- **Дихотомия:** сетки тривиальны (не нужен rebuild), графы/деревья — проблема не в rebuild стратегии, а в самом local update (он вносит структурный drift).

**Verdict:** CONDITIONAL PASS — проходит для grid, не проходит для graph/tree. Требуется переосмысление local update для нерегулярных пространств.

## Ключевые файлы

| Файл | Содержимое |
|------|-----------|
| `exp14_anchors.py` | Раннер эксперимента |
| `results/exp14_merged.json` | Объединённые результаты |
| `results/exp14_results.json` | Основные результаты |
| `results/exp14_graph_remaining.json` | Дополнительные прогоны для graph |
| `results/exp14_tree_only.json` | Дополнительные прогоны для tree |

---

# EN

## Problem

During incremental refinement the tree is updated locally (local update) without a full rebuild. Question: how much do local update results diverge from full rebuild? Which rebuild strategy minimizes divergence at minimal cost?

## Method

720 configurations: 4 spaces x 9 strategies x 20 seeds. 50 steps per config.

**Rebuild strategies:**
- `no_rebuild` -- local update only
- `periodic_{5,10,20,50}` -- full rebuild every K steps
- `dirty_{0.05,0.1,0.2,0.5}` -- rebuild when >X% nodes are dirty

## Kill Criteria and Results

**Kill criterion:** divergence < 5% (0.05).

| Space | Max Div | Mean Div | Best Strategy | Verdict |
|---|---|---|---|---|
| scalar_grid | 0.000 | 0.000 | no_rebuild | PASS |
| vector_grid | 0.000 | 0.000 | no_rebuild | PASS |
| irregular_graph | 1.517 | 0.374 | dirty_0.05 (0.204 mean) | FAIL |
| tree_hierarchy | 1.715 | 0.694 | dirty_0.05 (0.204 mean) | FAIL |

## Key Findings

- **Grids (scalar_grid, vector_grid):** divergence = 0.000 for ALL strategies. Local update is perfect -- rebuild not needed.
- **Graph and tree:** dirty-triggered (threshold 0.05) is the best strategy (mean_div=0.204), but still far from the 0.05 threshold. Periodic strategies are worse than dirty-triggered.
- **Dichotomy:** grids are trivial (no rebuild needed), graphs/trees -- the problem is not the rebuild strategy but local update itself (it introduces structural drift).

**Verdict:** CONDITIONAL PASS -- passes for grid, fails for graph/tree. Local update for irregular spaces requires redesign.

## Key Files

| File | Contents |
|------|----------|
| `exp14_anchors.py` | Experiment runner |
| `results/exp14_merged.json` | Merged results |
| `results/exp14_results.json` | Main results |
| `results/exp14_graph_remaining.json` | Additional graph runs |
| `results/exp14_tree_only.json` | Additional tree runs |
