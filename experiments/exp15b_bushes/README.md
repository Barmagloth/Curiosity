# Exp15b: Bush Clustering (Leaf-Path Clusters)

**Status:** FAIL (80 configs, Silhouette PASS but ARI FAIL)
**Roadmap level:** P3b

---

# RU

## Проблема

Существуют ли естественные кластеры среди leaf-путей в дереве уточнений? Если да и они стабильны между seeds -- leaf-path структура несёт семантическую информацию.

## Метод

80 конфигураций: 4 пространства x 20 seeds.

**Методы кластеризации:**
- k-means (k=2..10)
- DBSCAN (sweep по eps)
- Agglomerative clustering

Оценка: Silhouette score (качество кластеров внутри seed) и ARI (стабильность между seeds).

## Kill Criteria и результаты

**Kill criterion:** Silhouette > 0.4 AND cross-run ARI > 0.6.

| Пространство | Silhouette (mean +/- std) | k_mode | ARI | Вердикт |
|---|---|---|---|---|
| scalar_grid | 0.661 +/- 0.125 | 2 | 0.073 | FAIL |
| vector_grid | 0.793 +/- 0.065 | 2 | 0.094 | FAIL |
| irregular_graph | 0.649 +/- 0.082 | 2 | -0.011 | FAIL |
| tree_hierarchy | 0.485 +/- 0.067 | 2 | 0.210 | FAIL |

## Ключевые находки

- Silhouette проходит порог 0.4 во всех пространствах -- кластеры внутри отдельного seed визуально осмысленны.
- ARI катастрофически низкий -- кластеры НЕ воспроизводятся между seeds.
- **Вывод:** кластеры leaf-путей -- артефакт конкретного seed, а не устойчивое свойство пространства.

**Verdict:** FAIL. Silhouette > 0.4 PASS, но ARI stability FAIL. Bushes не стабильны.

### Заметки для revisit

- Кластеры существуют внутри каждого seed (Silhouette > 0.4). Нестабильность между seeds может означать зависимость от конкретного GT, а не бесполезность метода.
- Leaf-path similarity -- потенциальный инструмент для обнаружения дублирующих регионов и merge candidates.
- Связь с trajectory profiles (exp16): trajectory profiles кластеризуются стабильнее (Gap > 1.0). Запланирован revisit после первых результатов Track C.

## Ключевые файлы

| Файл | Содержимое |
|------|-----------|
| `exp15b_bushes.py` | Раннер эксперимента |
| `results/exp15b_results.json` | Основные результаты |
| `results/exp15b_stability.json` | Результаты ARI-стабильности |

---

# EN

## Problem

Do natural clusters exist among leaf-paths in the refinement tree? If yes and they are stable across seeds -- leaf-path structure carries semantic information.

## Method

80 configurations: 4 spaces x 20 seeds.

**Clustering methods:**
- k-means (k=2..10)
- DBSCAN (eps sweep)
- Agglomerative clustering

Evaluation: Silhouette score (within-seed cluster quality) and ARI (cross-seed stability).

## Kill Criteria and Results

**Kill criterion:** Silhouette > 0.4 AND cross-run ARI > 0.6.

| Space | Silhouette (mean +/- std) | k_mode | ARI | Verdict |
|---|---|---|---|---|
| scalar_grid | 0.661 +/- 0.125 | 2 | 0.073 | FAIL |
| vector_grid | 0.793 +/- 0.065 | 2 | 0.094 | FAIL |
| irregular_graph | 0.649 +/- 0.082 | 2 | -0.011 | FAIL |
| tree_hierarchy | 0.485 +/- 0.067 | 2 | 0.210 | FAIL |

## Key Findings

- Silhouette exceeds the 0.4 threshold in all spaces -- clusters within a single seed are visually meaningful.
- ARI is catastrophically low -- clusters do NOT reproduce across seeds.
- **Conclusion:** leaf-path clusters are an artifact of a specific seed, not a stable property of the space.

**Verdict:** FAIL. Silhouette > 0.4 PASS, but ARI stability FAIL. Bushes are not stable.

### Notes for Revisit

- Clusters exist within each seed (Silhouette > 0.4). Cross-seed instability may indicate dependence on specific GT, not method uselessness.
- Leaf-path similarity is a potential tool for detecting duplicate regions and merge candidates.
- Connection with trajectory profiles (exp16): trajectory profiles cluster more stably (Gap > 1.0). Revisit planned after first Track C results.

## Key Files

| File | Contents |
|------|----------|
| `exp15b_bushes.py` | Experiment runner |
| `results/exp15b_results.json` | Main results |
| `results/exp15b_stability.json` | ARI stability results |
