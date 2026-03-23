# Архитектура Curiosity

Документ описывает ключевые компоненты системы и принятые архитектурные решения. Каждое решение подкреплено результатами экспериментов.

---

## Общая схема

```
Пространство X (произвольной природы)
       |
       v
[Coarse representation] — грубое начальное приближение
       |
       v
[Функция информативности ρ(x)] — определяет, где уточнять
       |
       v
[Split decision] — решение о дроблении региона
       |
       v
[Adaptive refinement] — уточнение выбранных регионов
       |
       v
[Scale-Consistency check] — D_parent < τ_parent? (v1.7)
       |
       v
[Halo blending] — cosine feathering на границах
       (если применим по правилу топологии)
       |
       v
[Budget governor] — EMA-контроль расхода бюджета
       |
       v
[Probe allocation] — 5–10% бюджета на exploration
```

---

## Компонент 1: Функция информативности ρ(x)

### Сигналы

| Сигнал | Назначение | Слабость |
|---|---|---|
| Residual | Основной: ошибка текущей аппроксимации | Деградирует при шуме (corr 0.90->0.54) |
| HF energy | Лапласиан/градиент, структурная энергия | Ложные срабатывания на стыках |
| Variance | Локальная дисперсия / disagreement | Любит шум |
| Окупаемость | Expected gain vs. стоимость | — |

### Двухстадийный гейт (canonical решение)

**Stage 1:** Проверка здоровья residual
- Метрики: instability, FSR (False Signal Rate)
- Если здоров: ρ = residual-only (нулевые потери на clean данных)
- Если нездоров: переход к Stage 2

**Stage 2:** Utility-weighted комбинация
- Веса сглажены EMA с гистерезисом
- Нормализация: квантильная (ранговая), не абсолютная

### Three-layer rho decomposition (Phase 3.5, exp17)

ρ декомпозируется на три архитектурных слоя:

| Слой | Назначение | Частота пересчёта |
|------|-----------|-------------------|
| L0 (topology) | Структурные свойства пространства (кривизна, связность) | Редко (при перестройке дерева) |
| L1 (presence) | Каскадные квоты, dirty-сигнатуры | Каждый step (инкрементально) |
| L2 (query) | Конкретный запрос (residual, HF, utility-weighted комбинация) | Каждый query |

Cascade quotas (Variant C) управляют бюджетом по слоям. Streaming pipeline обеспечивает переиспользование L0/L1 между запросами. Reusability валидирована 12/12 PASS (min 0.838) на 1080 конфигурациях.

---

## Компонент 2: Halo (boundary-aware blending)

**Зачем:** Жёсткая вставка refined-тайла создаёт ступеньку на границе. Лапласиан ловит это как ложный HF-сигнал. Чем умнее adaptive выбор, тем сильнее артефакт.

**Реализация:**
- Overlap ≥ 3 элемента (не "пикселя" — система не привязана к изображениям)
- Cosine feathering относительно coarse уровня

**Правило применимости (Phase 0, concept v1.7 §6):**

Halo применим ТОЛЬКО когда выполнены ОБА условия:
1. **Boundary parallelism ≥ 3** — минимум 3 независимых cross-edge на границе тайлов
2. **No context leakage** — halo expansion не протекает в несвязанные тайлы

| Топология | Halo? | Причина |
|---|---|---|
| Grid (tile ≥ 3) | ✅ Всегда | boundary = широкая полоса, нет leakage |
| k-NN graph | ✅ Применять | min_cut обычно >> 3 |
| Tree / Forest | ❌ Никогда | min_cut=1 (bottleneck), context leakage в sibling-поддеревья |
| DAG | ⚠️ Per-case | Проверять boundary parallelism и leakage |

Root cause tree failure: single-edge bottleneck + sibling bleed (85% fade на hop 1 достигает чужого поддерева) + extreme S/V asymmetry (0.032 vs 0.5 для grid).

**Примечание:** свойства «инициализация нулём», «ограничение по энергии», «при отключении уровня — валидный откат» относятся к delta / уровню уточнения в целом, а не к halo. Halo — механизм согласования границ между тайлами, а не самостоятельный residual-носитель.

---

## Компонент 3: Probe (exploration)

**Зачем:** Без exploration система становится структурно слепой. Exploitation-only пропускает сдвиги, редкие паттерны, новые области интереса.

**Бюджет:** 5–10% от общего бюджета на каждом шаге.

**Приоритеты probe:**
1. Coarse residual / variance
2. Неопределённость (uncertainty)
3. Давность последней проверки

**Trade-off:** На стационарных сценах probe может слегка снижать PSNR, но это страховка от слепоты, а не оптимизация качества.

---

## Компонент 4: Budget Governor (EMA-контроллер)

**Зачем:** Без governor'а бюджет — декларация, а не ограничение.

**Объект управления:** strictness — квантильный порог отбора кандидатов.

**Параметры:**
- Δstrictness ≤ clamp (anti-oscillation)
- hard_cap_per_step = 3× target (safety fuse)
- Warmup: N шагов без движения strictness
- Compliance: асимметричная (overbudget penalty > underbudget)

**Числа (Exp0.8v5):**
- StdCost: −50% (~5.15 -> ~3.25)
- P95: 11.0 -> ~6.5
- Compliance penalty: −85% (~3.2 -> ~0.5)
- PSNR: −0.24 дБ (clean), −0.68 дБ (shift) — незначительная плата

---

## Компонент 5: SeamScore — метрика швов

**Формула:** `SeamScore = Jumpout / (Jumpin + eps)`

Вычисляется по edge strips (полоскам на границе тайлов).

**Валидация:** 2D scalar grids, vector-valued grids, irregular graphs, tree hierarchies.

**Статус:** Валидирована и стабильна в текущем scope валидации (4 типа пространств). Финальная production-readiness зависит от результатов P0–P4.

---

## Компонент 6: Scale-Consistency Invariant (v1.7)

**Зачем:** Halo обеспечивает локальную геометрическую корректность на границах тайлов. Но refined уровень может быть гладким по швам и при этом семантически противоречить coarse уровню — delta «контрабандой» проталкивает низкочастотный смысл наверх. Scale-Consistency Invariant гарантирует, что этого не происходит.

**Принцип:** coarse — якорь, delta — подчинённая поправка. Delta должна быть невидима с уровня выше.

**Формальное требование:**
```
‖R(delta)‖ / (‖delta‖ + ε) < τ_rel
```

**Пара операторов (R, Up):**
- **R** (coarse-graining): `gaussian blur (σ=3.0) + decimation`. Проецирует fine → coarse.
- **Up** (восстановление): `bilinear upsampling`. Проекция обратно в fine. Не обратный оператор к R.
- Пара фиксируется до экспериментов. Разные пары дают разные физики дерева.

**Метрики:**

| Метрика | Формула | Интерпретация |
|---|---|---|
| D_parent | `‖R(delta)‖ / (‖delta‖ + ε)` | Доля LF-энергии в delta. Ниже = лучше. Enforcement-сигнал. |
| D_hf | `‖delta - Up(R(delta))‖ / (‖delta‖ + ε)` | HF-чистота delta. Выше = лучше. Диагностика, не hard constraint. |

**Enforcement (после baseline-валидации):**
- `D_parent > τ_parent` → damp delta / reject split / increase local strictness
- D_parent также используется как контекстный сигнал в ρ (не самодостаточный)

**Пороги:** τ_parent — data-driven по baseline-эксперименту, может зависеть от уровня L.

**Кросс-пространственная валидация (Phase 0):**

| Пространство | D_parent AUC | D_hf AUC |
|---|---|---|
| T1 Scalar grid | 1.000 | 0.806 |
| T2 Vector grid | 1.000 | 0.810 |
| T3 Irregular graph | 1.000 | — |
| T4 Tree hierarchy | 0.824 | — |

**Статус:** SC-baseline (SC-0..SC-4) ✅ ЗАВЕРШЁН. D_parent валидирован на 4 пространствах. SC-5 (τ_parent) и SC-enforce — открыты. Протокол: `scale_consistency_verification_protocol_v1.0.md`.

---

## Компонент 7: Дерево уточнения

Дерево — журнал маршрутов решений split. Каждый путь от корня к листу = последовательность решений.

**Требования:**
- GPU-дружественная структура (плоская упаковка, без pointer chasing)
- Morton и block-sparse layouts: **отложены (deferred)** по итогам Exp0.9a microbench (sort overhead / expansion ratio). Окончательное решение — после P0 (0.9b0+).
- Compact layouts: предварительно перспективны при low sparsity + large grid. Kill/go — в Exp0.9b0.

**Куст** = множество путей, ведущих к одному смыслу. Метрика расстояния через LCA / общий префикс.

---

## Компонент 8: Graph Community Detection

**Задача:** Разбиение нерегулярного графа (T3: irregular_graph) на связные community для формирования refinement units.

**Требования:**
1. Кластеры должны быть **топологически связными** — R/Up операторы (restrict = cluster-mean, prolong = piecewise constant) требуют этого для корректности.
2. Разбиение должно следовать **структуре связности**, а не геометрии координат. На скрученных манифольдах (Swiss Roll и подобные) геометрия лжёт.
3. Сложность O(N log N) или лучше — линейная масштабируемость по числу узлов.

**Primary: Leiden** (`leidenalg` + `igraph`)
- Максимизирует модулярность, режет строго по топологическим швам.
- Нативная гарантия связности community (каждый кластер = связный подграф).
- O(N log N), детерминистичен при фиксированном seed.
- Edge cut 6.9–7.9% на Swiss Roll тестах (vs 12.7–15.3% у k-means).
- D_parent на 14–16% ниже чем у k-means на патологических графах.

**Fallback: Louvain + CC post-fix** (NetworkX, zero C-dependencies)

На платформах, где `igraph` не может быть скомпилирован (экзотические ARM-архитектуры, минимальные контейнеры без build tools, среды без C-компилятора), допустима замена на связку:
1. `networkx.algorithms.community.louvain_communities()` — O(N log N), чистый Python.
2. `networkx.connected_components()` на каждом кластере — O(V+E), линейное время.
3. Если внутри кластера обнаружены изолированные компоненты — каждый получает уникальный ID.

Результат функционально идентичен Leiden: связные community, топологическое разбиение. Fallback активируется автоматически при `ImportError` на `igraph`/`leidenalg`.

**Ограничения fallback:** NetworkX Louvain в ~50–80× медленнее и потребляет ~70–170× больше памяти, чем Leiden (замеры на PC2, RTX 2070). На больших графах (>10K узлов) это может быть неприемлемо. Если Leiden недоступен на целевом железе — рекомендуется решить проблему на уровне деплоя (установить C-компилятор), а не мириться с производительностью fallback.

**Валидация:** `experiments/exp_phase2_pipeline/test_swiss_roll.py` — Swiss Roll stress test, 4 конфигурации (500–3000 точек), 3 seed, 4 алгоритма (k-means, spectral, Louvain, Leiden).

| Метрика | k-means | Spectral | Louvain | Leiden |
|---|---|---|---|---|
| Edge Cut (pathological) | 12.7% | 10.6% | 7.2% | **6.9%** |
| D_parent mean | 0.098 | 0.099 | 0.085 | **0.082** |
| Связность | ❌ | ❌ | ✅ (post-fix) | ✅ (native) |

---

## Стек технологий

- **Язык:** Python
- **ML-фреймворк:** PyTorch
- **GPU:** CUDA (environment_2) / DirectML (environment_1, AMD GPU)
- **Окружения:** `.venv` (CPU, Python 3.13) + `.venv-gpu` (DirectML, Python 3.12). См. `docs/environment_1.md`
- **Эксперименты:** Jupyter Notebooks
- **Документация:** Versioned markdown
- **Зависимости:** `requirements.txt` в корне проекта
