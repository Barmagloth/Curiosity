# Exp16: C-pre -- Trajectory Profile Clustering

**Status:** PASS (80 configs, all 4 spaces pass both thresholds) -- Track C UNFREEZE
**Roadmap level:** C-pre (Phase 3, after S1-S3)

---

# RU

## Проблема

Следуют ли refinement units дискретным поведенческим профилям (trajectory profiles) во времени? Например: "всегда проходит гейт легко" vs "всегда заглушается/отклоняется". Если да -- существует кластерная структура, Track C можно разморозить.

## Метод

80 конфигураций: 4 пространства x 20 seeds.

**Trajectory features:** EMA-квантили, split signatures.

**Оценка:** Gap statistic + Silhouette + ARI (cross-seed stability).

## Kill Criteria и результаты

**Kill criterion:** Gap > 1.0 AND Silhouette > 0.3.

| Пространство | Gap (mean +/- std) | Silhouette (mean +/- std) | k_mode | ARI | Вердикт |
|---|---|---|---|---|---|
| scalar_grid | 1.37 +/- 0.09 | 0.605 +/- 0.148 | 4 | 0.454 | PASS |
| vector_grid | 2.04 +/- 0.26 | 0.794 +/- 0.125 | 4 | 0.086 | PASS |
| irregular_graph | 1.39 +/- 0.41 | 0.518 +/- 0.117 | 2 | 0.029 | PASS |
| tree_hierarchy | 2.40 +/- 1.01 | 0.453 +/- 0.075 | 7 | 0.441 | PASS |

## Ключевые находки

- Все 4 пространства проходят оба порога (gap > 1.0, silhouette > 0.3).
- ARI умеренный для scalar_grid (0.454) и tree_hierarchy (0.441), низкий для vector_grid и irregular_graph -- кластеры реальны (высокий gap), но границы нестабильны.
- k_mode варьируется: 2-4 для grid/graph, 7 для tree -- разные пространства дают разное количество профилей.

**Verdict:** PASS. Track C **UNFREEZE**. Trajectory features демонстрируют реальную кластерную структуру.

## Ключевые файлы

| Файл | Содержимое |
|------|-----------|
| `exp16_cpre.py` | Раннер эксперимента |
| `results/exp16_results.json` | Основные результаты (gap, silhouette, k) |
| `results/exp16_stability.json` | Результаты ARI-стабильности |

---

# EN

## Problem

Do refinement units follow discrete behavioral profiles (trajectory profiles) over time? For example: "always passes gate easily" vs "always damped/rejected". If yes -- cluster structure exists and Track C can be unfrozen.

## Method

80 configurations: 4 spaces x 20 seeds.

**Trajectory features:** EMA quantiles, split signatures.

**Evaluation:** Gap statistic + Silhouette + ARI (cross-seed stability).

## Kill Criteria and Results

**Kill criterion:** Gap > 1.0 AND Silhouette > 0.3.

| Space | Gap (mean +/- std) | Silhouette (mean +/- std) | k_mode | ARI | Verdict |
|---|---|---|---|---|---|
| scalar_grid | 1.37 +/- 0.09 | 0.605 +/- 0.148 | 4 | 0.454 | PASS |
| vector_grid | 2.04 +/- 0.26 | 0.794 +/- 0.125 | 4 | 0.086 | PASS |
| irregular_graph | 1.39 +/- 0.41 | 0.518 +/- 0.117 | 2 | 0.029 | PASS |
| tree_hierarchy | 2.40 +/- 1.01 | 0.453 +/- 0.075 | 7 | 0.441 | PASS |

## Key Findings

- All 4 spaces pass both thresholds (gap > 1.0, silhouette > 0.3).
- ARI is moderate for scalar_grid (0.454) and tree_hierarchy (0.441), low for vector_grid and irregular_graph -- clusters are real (high gap) but boundaries are unstable.
- k_mode varies: 2-4 for grid/graph, 7 for tree -- different spaces yield different numbers of profiles.

**Verdict:** PASS. Track C **UNFREEZE**. Trajectory features demonstrate real cluster structure.

## Key Files

| File | Contents |
|------|----------|
| `exp16_cpre.py` | Experiment runner |
| `results/exp16_results.json` | Main results (gap, silhouette, k) |
| `results/exp16_stability.json` | ARI stability results |
