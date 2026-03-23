# Exp18: Basin Membership

**Status:** DONE (80 configs, 0 errors, ALL FAIL)
**Roadmap level:** Phase 3 follow-up (deferred post-Phase 4)

---

# RU

## Гипотеза

Дерево = RG-flow. Правильная метрика семантичности — basin membership (принадлежность к одному бассейну аттрактора), а не LCA-distance (которая провалилась в exp15). Юниты, сходящиеся к одной неподвижной точке (rho_final/rho_initial < 0.10), должны быть ближе в feature space, чем юниты из разных бассейнов.

## Метод

1. Построить дерево через pipeline (single-pass, budget=0.30).
2. Определить бассейны — группы листьев, сходящихся к одной неподвижной точке (критерий: rho_final/rho_initial < 0.10).
3. Вычислить point-biserial корреляцию между same_basin (бинарная переменная: 1 если оба юнита в одном бассейне, 0 иначе) и feature_similarity (непрерывная).

## Результаты

| Пространство | pb_r | n_basins | same_basin% | Вердикт |
|-------------|------|----------|-------------|---------|
| scalar_grid | 0.023 | 56/64 | 0.9% | FAIL |
| vector_grid | 0.000 | 16/16 | 0.0% | FAIL |
| irregular_graph | 0.053 | 8.7/10 | 4.6% | FAIL |
| tree_hierarchy | 0.000 | 8/8 | 0.0% | FAIL |

## Kill criteria

Point-biserial r > 0.3: **ALL FAIL** (overall r = 0.019).

## Почему провалилось

При 30% бюджете в single-pass юниты не достигают неподвижных точек. Бассейны вырождаются в синглтоны (почти каждый юнит — свой собственный "бассейн"). Нужен multi-pass для достаточно глубоких деревьев, где RG-flow успевает сойтись.

## Отложено

Вернуться после Phase 4+ (после реализации multi-pass). Пока multi-pass не реализован, basin membership не имеет смысла проверять.

## Ключевые файлы

| Файл | Содержимое |
|------|-----------|
| `exp18_basin_membership.py` | Раннер эксперимента |
| `results/exp18_results.json` | Полные результаты |
| `results/exp18_summary.json` | Агрегированная сводка |

---

# EN

## Hypothesis

Tree = RG-flow. The correct semantics metric is basin membership (same fixed-point attractor), not LCA-distance (which failed in exp15). Units converging to the same fixed point (rho_final/rho_initial < 0.10) should correlate with feature similarity better than units from different basins.

## Method

1. Build tree via pipeline (single-pass, budget=0.30).
2. Identify basins — groups of leaves converging to the same fixed point (criterion: rho_final/rho_initial < 0.10).
3. Compute point-biserial correlation between same_basin (binary: 1 if both units share a basin, 0 otherwise) and feature_similarity (continuous).

## Results

| Space | pb_r | n_basins | same_basin% | Verdict |
|-------|------|----------|-------------|---------|
| scalar_grid | 0.023 | 56/64 | 0.9% | FAIL |
| vector_grid | 0.000 | 16/16 | 0.0% | FAIL |
| irregular_graph | 0.053 | 8.7/10 | 4.6% | FAIL |
| tree_hierarchy | 0.000 | 8/8 | 0.0% | FAIL |

## Kill Criteria

Point-biserial r > 0.3: **ALL FAIL** (overall r = 0.019).

## Why It Failed

At 30% budget in single-pass, units do not reach fixed points. Basins degenerate to singletons (nearly every unit is its own "basin"). Needs multi-pass for deep enough trees where RG-flow has time to converge.

## Deferred

Revisit post-Phase 4 (after multi-pass is implemented). Until multi-pass exists, basin membership is not meaningfully testable.

## Key Files

| File | Contents |
|------|----------|
| `exp18_basin_membership.py` | Experiment runner |
| `results/exp18_results.json` | Full results |
| `results/exp18_summary.json` | Aggregated summary |
