# Exp15: LCA-Distance vs Feature Similarity

**Status:** FAIL (80 configs, max Spearman 0.299 < 0.3 threshold)
**Roadmap level:** P3a

---

# RU

## Проблема

Коррелирует ли LCA-расстояние в дереве уточнений с семантической близостью (||feature_i - feature_j||)? Если да -- дерево является семантической картой, а не просто журналом вычислений.

## Метод

80 конфигураций: 4 пространства x 20 seeds. 500 пар юнитов на каждый seed.

Для каждой пары юнитов вычисляется:
- LCA-расстояние (глубина наименьшего общего предка)
- Евклидово расстояние между feature-векторами

Оценивается корреляция Спирмена и Пирсона.

## Kill Criteria и результаты

**Kill criterion:** Spearman r > 0.3.

| Пространство | Spearman (mean +/- std) | Pearson (mean +/- std) | Вердикт |
|---|---|---|---|
| scalar_grid | 0.299 +/- 0.108 | 0.283 +/- 0.087 | FAIL (0.299 < 0.3) |
| vector_grid | -0.032 +/- 0.035 | -0.031 +/- 0.024 | FAIL |
| irregular_graph | 0.267 +/- 0.113 | 0.272 +/- 0.110 | FAIL |
| tree_hierarchy | 0.006 +/- 0.064 | 0.079 +/- 0.072 | FAIL |

## Ключевые находки

- scalar_grid ближе всего к порогу (0.299), отдельные seeds до 0.49, но высокая дисперсия.
- vector_grid и tree_hierarchy -- практически нулевая корреляция.
- **Вывод:** LCA-расстояние не является надёжной метрикой семантической близости. Дерево -- журнал уточнений, а не семантическая карта.

**Verdict:** FAIL. Дерево не семантично по критерию LCA-distance.

## Ключевые файлы

| Файл | Содержимое |
|------|-----------|
| `exp15_lca_distance.py` | Раннер эксперимента |
| `results/exp15_results.json` | Результаты |

---

# EN

## Problem

Does LCA-distance in the refinement tree correlate with feature similarity (||feature_i - feature_j||)? If yes -- the tree is a semantic map, not just a computation journal.

## Method

80 configurations: 4 spaces x 20 seeds. 500 unit pairs per seed.

For each pair of units:
- LCA-distance (depth of the lowest common ancestor)
- Euclidean distance between feature vectors

Spearman and Pearson correlations are evaluated.

## Kill Criteria and Results

**Kill criterion:** Spearman r > 0.3.

| Space | Spearman (mean +/- std) | Pearson (mean +/- std) | Verdict |
|---|---|---|---|
| scalar_grid | 0.299 +/- 0.108 | 0.283 +/- 0.087 | FAIL (0.299 < 0.3) |
| vector_grid | -0.032 +/- 0.035 | -0.031 +/- 0.024 | FAIL |
| irregular_graph | 0.267 +/- 0.113 | 0.272 +/- 0.110 | FAIL |
| tree_hierarchy | 0.006 +/- 0.064 | 0.079 +/- 0.072 | FAIL |

## Key Findings

- scalar_grid is closest to the threshold (0.299), individual seeds reach 0.49, but high variance.
- vector_grid and tree_hierarchy show near-zero correlation.
- **Conclusion:** LCA-distance is not a reliable measure of semantic similarity. The tree is a refinement journal, not a semantic map.

**Verdict:** FAIL. The tree is not semantic by LCA-distance criterion.

## Key Files

| File | Contents |
|------|----------|
| `exp15_lca_distance.py` | Experiment runner |
| `results/exp15_results.json` | Results |
