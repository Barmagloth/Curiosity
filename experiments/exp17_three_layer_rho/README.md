# Exp17: Three-Layer rho Decomposition

**Status:** DONE (1080 configs, 0 errors)
**Roadmap level:** Phase 4 prerequisite

---

# RU

## Проблема

Монолитная функция rho смешивает три ответственности:
1. **Пространственная структура** -- как разбить домен на регионы (кластеры, квадранты, поддеревья).
2. **Присутствие данных** -- где данные реально есть, а где пустые области.
3. **Задачно-специфичное уточнение** -- какой запрос мы оптимизируем (MSE, max ошибка, остаточная частота).

Из-за этого дерево уточнения (refinement tree) нужно строить заново при смене запроса, повторно использовать слои невозможно, а отладка непрозрачна: непонятно, какой именно аспект rho вызывает артефакт.

## Архитектура

Три каскадных слоя, каждый следующий уточняет результат предыдущего:

### Layer 0 -- Topology (независим от данных)

Чистый структурный анализ домена. Выход: дерево кластеров с приоритетами.

| Тип домена | Метод |
|------------|-------|
| Граф | Leiden-кластеры + Ollivier--Ricci кривизна + PageRank + граничные аномалии |
| Дерево | Оценка по глубине (depth-based scoring) |
| Регулярная сетка | Разбиение на пространственные квадранты (тривиальная топология) |

Так как L0 не зависит от данных, его результат можно вычислить один раз и переиспользовать для любых наборов данных на том же домене.

### Layer 1 -- Presence (зависит от данных, не зависит от запроса)

Оценивает, где в доменной структуре реально присутствуют значимые данные. Метрика: дисперсия ground truth внутри каждого юнита.

**Cascade Quotas (Вариант C):** Каждый L0-кластер гарантирует минимум выживших юнитов, пропорционально размеру кластера:

```
K = max(1, ceil(cluster_size * 0.3))
```

Это решает критическую проблему: фиксированный порог (`l1_threshold=0.01`) убивал 97% юнитов на `scalar_grid` при масштабе 1000. С каскадными квотами ни один регион не вымирает полностью.

**До и после квот:**
- Без квот: reusability ratio = 0.725 (FAIL, порог 0.80)
- С квотами: reusability ratio = 0.928 (PASS)

### Layer 2 -- Query (задачно-специфичный)

Три взаимозаменяемые query-функции работают на одном и том же замороженном дереве (FrozenTree):

| Функция | Описание |
|---------|----------|
| `mse` | Среднеквадратичная ошибка -- стандартный выбор |
| `max_abs` | Максимальная абсолютная ошибка -- для worst-case гарантий |
| `hf_residual` | Высокочастотный остаток -- для обнаружения осцилляций |

Ключевое свойство: смена query-функции **не** требует пересчёта L0 и L1.

## Потоковый конвейер (Streaming Pipeline)

Вместо последовательной обработки всех юнитов (L0(all) -> L1(all) -> L2(all)), конвейер обрабатывает кластер за кластером:

1. Кластеры упорядочиваются по L0-приоритету.
2. Для каждого кластера последовательно применяются L0 -> L1 -> L2.
3. Глобальный бюджет (budget cap) останавливает обработку, когда достигнут лимит.

Результат: 10-20% быстрее batch-режима на регулярных сетках за счёт раннего отсечения.

## Kill Criteria и результаты

| Критерий | Метрика | Порог | Результат |
|----------|---------|-------|-----------|
| Переиспользуемость дерева | min(psnr_frozen / psnr_fresh) | >= 0.80 | 12/12 PASS (min ratio 0.838) |
| Амортизированный break-even | Запросов до 3-layer < single_pass * N | <= 5 | PASS |
| Накладные L0 | l0_ms / topo_profiling_ms | < 2.0x | PASS |
| Ускорение L2 | l2_ms / single_pass_ms | < 0.50x | PASS |

### Нюансы по PSNR

- **Сетки:** на 2-4 дБ ниже single_pass -- цена L1-прунинга (каскадные квоты сохраняют структуру, но жертвуют точностью).
- **Графы/деревья:** паритет с single_pass.
- **Индустриальное сравнение:** `scipy.spatial.cKDTree` (C-реализация) быстрее на одиночном запросе, но не поддерживает side data и не амортизируется при множественных запросах.

### Масштаб эксперимента

- 1080 конфигураций
- 0 ошибок
- Все конфигурации прошли без исключений

## Ключевые файлы

| Файл | Содержимое |
|------|-----------|
| `layers.py` | `Layer0_Topology`, `Layer1_Presence`, `Layer2_Query`, `FrozenTree`, `ThreeLayerPipeline`, `IndustryBaselines` |
| `exp17_three_layer_rho.py` | Раннер эксперимента с поддержкой `--chunk` для инкрементального запуска |
| `config17.py` | Конфигурация эксперимента (все 1080 комбинаций) |
| `results/` | JSON-файлы результатов по чанкам |

## Дорожная карта

Оптимизация scoring-фаз на C/Cython даст мультипликативное ускорение для потокового режима:

| Фаза | Текущее | Целевое |
|------|---------|---------|
| L0 topology | 70 мс | ~5 мс |
| L1 + L2 scoring | 13 мс | ~1.3 мс |

Подробности: `docs/workplan.md`, секция H.

---

# EN

## Problem

The monolithic rho function mixes three concerns:
1. **Spatial structure** -- how to partition the domain into regions (clusters, quadrants, subtrees).
2. **Data presence** -- where data actually exists vs. empty regions.
3. **Task-specific refinement** -- which query we are optimizing (MSE, max error, high-frequency residual).

This forces a full refinement tree rebuild on every query change, prevents layer reuse, and makes debugging opaque: it is unclear which aspect of rho causes an artifact.

## Architecture

Three cascading layers, each refining the output of the previous one:

### Layer 0 -- Topology (data-independent)

Pure structural analysis of the domain. Output: a cluster tree with priorities.

| Domain type | Method |
|-------------|--------|
| Graph | Leiden clusters + Ollivier--Ricci curvature + PageRank + boundary anomalies |
| Tree | Depth-based scoring |
| Regular grid | Spatial quadrant blocks (trivial topology) |

Since L0 is data-independent, its result can be computed once and reused for any dataset on the same domain.

### Layer 1 -- Presence (data-dependent, query-independent)

Evaluates where meaningful data actually exists within the domain structure. Metric: variance of ground truth per unit.

**Cascade Quotas (Variant C):** Each L0 cluster guarantees a minimum number of survivors proportional to cluster size:

```
K = max(1, ceil(cluster_size * 0.3))
```

This solves a critical problem: a fixed threshold (`l1_threshold=0.01`) killed 97% of units on `scalar_grid` at scale 1000. With cascade quotas, no region dies completely.

**Before and after quotas:**
- Without quotas: reusability ratio = 0.725 (FAIL, threshold 0.80)
- With quotas: reusability ratio = 0.928 (PASS)

### Layer 2 -- Query (task-specific)

Three interchangeable query functions operate on the same frozen tree (FrozenTree):

| Function | Description |
|----------|-------------|
| `mse` | Mean squared error -- standard choice |
| `max_abs` | Maximum absolute error -- for worst-case guarantees |
| `hf_residual` | High-frequency residual -- for oscillation detection |

Key property: switching the query function does **not** require recomputing L0 and L1.

## Streaming Pipeline

Instead of processing all units sequentially (L0(all) -> L1(all) -> L2(all)), the pipeline processes cluster-by-cluster:

1. Clusters are ordered by L0 priority.
2. For each cluster, L0 -> L1 -> L2 are applied in sequence.
3. A global budget cap stops processing when the limit is reached.

Result: 10-20% faster than batch mode on regular grids due to early cutoff.

## Kill Criteria and Results

| Criterion | Metric | Threshold | Result |
|-----------|--------|-----------|--------|
| Tree reusability | min(psnr_frozen / psnr_fresh) | >= 0.80 | 12/12 PASS (min ratio 0.838) |
| Amortized break-even | Queries until 3-layer < single_pass * N | <= 5 | PASS |
| L0 overhead | l0_ms / topo_profiling_ms | < 2.0x | PASS |
| L2 speedup | l2_ms / single_pass_ms | < 0.50x | PASS |

### PSNR Nuances

- **Grids:** 2-4 dB below single_pass -- the cost of L1 pruning (cascade quotas preserve structure but sacrifice precision).
- **Graphs/trees:** parity with single_pass.
- **Industry comparison:** `scipy.spatial.cKDTree` (C implementation) is faster on a single query, but does not support side data and does not amortize across multiple queries.

### Experiment Scale

- 1080 configurations
- 0 errors
- All configurations completed without exceptions

## Key Files

| File | Contents |
|------|----------|
| `layers.py` | `Layer0_Topology`, `Layer1_Presence`, `Layer2_Query`, `FrozenTree`, `ThreeLayerPipeline`, `IndustryBaselines` |
| `exp17_three_layer_rho.py` | Experiment runner with `--chunk` support for incremental execution |
| `config17.py` | Experiment configuration (all 1080 combinations) |
| `results/` | Per-chunk JSON result files |

## Roadmap

C/Cython optimization of scoring phases would give multiplicative speedup for streaming:

| Phase | Current | Target |
|-------|---------|--------|
| L0 topology | 70 ms | ~5 ms |
| L1 + L2 scoring | 13 ms | ~1.3 ms |

Details: `docs/workplan.md`, section H.
