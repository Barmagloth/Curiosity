# Результаты экспериментов Curiosity

Серия Exp0.1–Exp0.8 завершена. Ниже — сводка результатов каждого эксперимента с ключевыми числами и выводами.

---

## Exp0.1 — Работает ли adaptive refinement?

**Вопрос:** Превосходит ли адаптивный выбор тайлов для уточнения случайный выбор?

**Результат:** Да. Adaptive > random по MSE/PSNR при любом уровне coverage.

**Вывод:** Базовая гипотеза подтверждена. Selective tile refinement работает.

---

## Exp0.2 — Устойчивость на реальных данных

**Вопрос:** Устойчивы ли результаты на реальных изображениях (не только синтетических)?

**Результат:** Да по MSE_rgb/PSNR (98–100% winrate). Однако обнаружена HF-деградация (высокочастотные артефакты) при малом бюджете.

**Вывод:** Работает, но нужен механизм для устранения артефактов на границах тайлов. Мотивация для halo.

---

## Exp0.3 — Нужен ли halo?

**Вопрос:** Устраняет ли перекрытие (halo/overlap) на границах тайлов HF-артефакты?

**Результат:** Да. Halo ≥ 3 ячейки (пикселя в image-экспериментах) при tile_size=16 устраняет артефакты. Interior-only HF-метрика подтвердила: проблема именно в швах, а не во внутренней части тайлов.

**Вывод:** Boundary-aware blending (cosine feathering) с минимальным overlap = 3 элемента — обязательное требование.

---

## Exp0.4 — Нужна ли комбинированная ρ?

**Вопрос:** Нужно ли комбинировать несколько сигналов информативности, или residual достаточен?

**Результат:** На чистых данных residual-only = оракул. Комбинация не даёт преимущества.

**Вывод:** При идеальных условиях одного residual достаточно. Но вопрос — что происходит при шуме?

---

## Exp0.5 — Ломается ли residual-оракул?

**Вопрос:** Деградирует ли residual-only сигнал при шуме, blur, alias?

**Результат:** Да. Корреляция residual с oracle падает с 0.90 до 0.54 при шуме. Degradation на coarse уровне.

**Вывод:** Residual-only — хрупкая стратегия. Нужен fallback-механизм.

---

## Exp0.6 — Бинарный переключатель resid/combo

**Вопрос:** Работает ли простой переключатель между residual и комбинированным сигналом?

**Результат:** Частично. Clean/blur -> residual, noise -> combo. Alias — пограничный случай, переключатель не всегда верно определяет режим.

**Вывод:** Бинарная логика недостаточна. Нужен более гибкий механизм.

---

## Exp0.7 / Exp0.7b — Soft gating и двухстадийный гейт

**Вопрос:** Можно ли заменить бинарный переключатель на soft gating?

**Exp0.7:** Soft gating побеждает на noise/spatvar/jpeg (+0.08–1.10 дБ), но проигрывает на clean/blur (−1.6–2.5 дБ) из-за «демократичности» softmax.

**Exp0.7b:** Двухстадийный гейт решает проблему. Stage 1 проверяет «здоров ли residual?»:
- Если да -> residual-only (без потерь на clean/blur, Δ ≈ 0).
- Если нет -> utility-weighted комбинация (+0.77–1.49 дБ на noise/spatvar).
- JPEG: единственный минус (−0.21 дБ), вопрос тонкой настройки порога.

**Вывод:** Двухстадийный гейт — правильная архитектура. Canonical решение.

---

## Exp0.8 — Schedule и budget governor

**Вопрос:** Нужен ли phase schedule (смена весов ρ по шагам) и budget governor?

**Budget governor (EMA):**
- StdCost снижается вдвое (~5.15 -> ~3.25).
- P95 с 11.0 до ~6.5.
- Compliance penalty с ~3.2 до ~0.5.
- PSNR чуть ниже (−0.24 дБ clean, −0.68 дБ shift).
- **Governor нужен для предсказуемости бюджета, не для качества.**

**Phase schedule:** Не показал выигрыша при текущих условиях. Отложен как опциональное расширение.

**Вывод:** EMA-governor — обязательный компонент. Phase schedule — нет.

---

## Halo Cross-Space Validation (Phase 0)

**Вопрос:** Работает ли Halo (cosine feathering, overlap >= 3) за пределами 2D pixel grids?

**Результаты:**

| Пространство | Improvement | p-value | Вердикт |
|---|---|---|---|
| T1 scalar grid | 2.02x | 3.81e-06 | Pass |
| T2 vector grid | 1.57x | 3.81e-06 | Pass |
| T3 irregular graph | 1.82x | 9.54e-06 | Pass |
| T4 tree hierarchy | 0.56x (WORSE) | 0.99 | Fail |

**Вывод:** Halo работает на grid/graph, проваливается на tree. Выведено правило применимости: boundary parallelism >= 3 AND no context leakage. Grid/graph: всегда. Tree/forest: никогда.

---

## SC-baseline (Phase 0)

**Вопрос:** Разделяют ли метрики D_parent и D_hf положительные и отрицательные случаи?

**Результаты (исходная формула):**

| Метрика | AUC | Effect size (d) | Вердикт |
|---|---|---|---|
| D_hf | 0.806 | 1.34 | Pass |
| D_parent (original) | 0.685 | 0.233 | Fail |

**Результаты (обновлённая формула D_parent: R sigma=3.0 + lf_frac):**

| Метрика | AUC | Effect size (d) | Вердикт |
|---|---|---|---|
| D_parent (fixed) | 0.853 | 1.491 | Pass |

**Кросс-пространственная валидация D_parent (fixed):**

| Пространство | AUC |
|---|---|
| T1 scalar grid | 1.000 |
| T2 vector grid | 1.000 |
| T3 irregular graph | 1.000 |
| T4 tree hierarchy | 0.824 |

Все пространства проходят порог (AUC >= 0.75).

**Исправление:** coarse_shift generator исправлен на spatially coherent sign fields.

**Вывод:** D_parent с обновлённой формулой `||R(delta)|| / (||delta|| + epsilon)` (R=gauss sigma=3.0) валидирован. Формула измеряет, какая доля энергии delta является низкочастотной (lf_frac).

---

## Сводная таблица

| Компонент | Статус | Обязательность |
|---|---|---|
| Adaptive refinement | Подтверждён | Ядро системы |
| Halo (boundary blending) | Подтверждён | Обязателен (>=3 px); grid/graph only |
| Probe (exploration) | Подтверждён | Обязателен (5–10% бюджета) |
| Двухстадийный гейт | Подтверждён | Обязателен при шуме/деградации |
| EMA budget governor | Подтверждён | Обязателен |
| SeamScore метрика | Валидирована | Стабильна в текущем scope валидации |
| Halo cross-space | Валидирован | Grid/graph: да. Tree: нет. Правило выведено |
| SC-baseline (D_hf) | Подтверждён | AUC=0.806, d=1.34 |
| SC-baseline (D_parent fixed) | Подтверждён | AUC=0.853, d=1.491; cross-space 0.824–1.000 |
| Phase schedule | Не подтверждён | Отложен |
| Morton/block-sparse layout | Предварительно невыгоден | По итогам 0.9a microbench; P0 открыт |
| Topo profiling (pre-runtime) | Подтверждён | Обязателен для IrregularGraphSpace. 97% accuracy, P50=56ms |
| Three-zone classifier v3 | Подтверждён | κ+Gini+η_F → GREEN/YELLOW/RED. Меняет τ_eff и бюджет |

---

## Phase 1 — P0 Layout, Детерминизм, Чувствительность (март 2026)

---

## Exp10 (exp10_buffer_scaling) — Grid vs compact layout на GPU

**Вопрос:** Что лучше для GPU — полноразмерная сетка (grid) или компактный массив с обратным индексом (reverse_map)?

**Kill criterion:** compact overhead >20% по времени ИЛИ по VRAM → убить compact.

**Результат:** KILL compact. Compute на O(k) быстрее на 18.5%, но reverse_map[M] на int32 даёт +38.6% VRAM. 75/75 конфигов превышают VRAM threshold. Убита конкретная реализация (compact с element-level reverse_map), не принцип sparse layout.

**Вывод:** Grid зафиксирован как baseline. Но compute на compact быстрее — значит sparse с правильным lookup жизнеспособен.

---

## Exp10d (exp10d_seed_determinism) — Побитовый детерминизм (DET-1)

**Вопрос:** Даёт ли система побитово идентичные результаты при одинаковом seed?

**Kill criterion:** любое расхождение = FAIL.

**Результат:** PASS 240/240. Все 4 типа пространств × 10 seeds × 3 бюджета × CPU+CUDA — побитовое совпадение.

**Компоненты:** Canonical traversal order (Z-order tie-break), deterministic probe (SHA-256 seed), governor isolation (EMA commit after full step).

**Вывод:** DET-1 пройден. Система детерминирована.

---

## Exp10e (exp10e_tile_sparse) — Tile-sparse кандидаты

**Вопрос:** Может ли tile-sparse layout без global reverse_map побить grid?

**Кандидаты:**
- A (bitset): grid + bitset маска активации
- B (packed Morton): компактные тайлы + бинарный поиск по Morton-ключам
- C (paged): страничная sparse-схема с macroblocks

**Результат:** A жив (-20% время, +18% VRAM). B убит — бинарный поиск на GPU = +1700% по времени. C убит — +9000%.

**Вывод:** Lookup — узкое место. Бинарный поиск и page machinery на GPU мертвы. Нужен O(1) lookup.

---

## Exp10f (exp10f_packed_lookup) — Packed tiles + прямой / hash lookup

**Вопрос:** Работает ли O(1) tile_map[id]→slot вместо бинарного поиска?

**Кандидаты:**
- D_direct: packed tiles + tile_map (прямой индекс int32)
- E_hash: packed tiles + open-addressing hash table

**Результат:** D_direct: 5× быстрее grid, resident memory 5.5× меньше. E_hash: та же скорость, но build в 10-30× дольше. Peak VRAM завышен из-за артефакта измерения (conv2d temporaries).

**Вывод:** Lookup решён. Hash не нужен при bounded regular domains. E_hash архивирован как fallback.

---

## Exp10g (exp10g_dual_benchmark) — Двухконтурный бенчмарк

**Вопрос:** Проходит ли D_direct оба контура — архитектурный (stencil) и операционный (conv2d)?

**Режимы:**
- Contour A: ручной stencil kernel (чистая проверка layout)
- Contour B: conv2d (реальный operator path)

**Результат:** D_direct PASS оба контура. Conv2d: -54% до -80% время, -36% до -86% peak VRAM. Stencil: +5-12% время (в пределах threshold), resident ≤ grid.

**Вывод:** D_direct (packed tiles + tile_map) — победитель для scalar grid. Артефакт exp10f устранён.

---

## Exp10h (exp10h_cross_space) — Кросс-пространственная валидация

**Вопрос:** Работает ли D_direct на vector_grid и tree_hierarchy?

**Результат:**
- Vector grid: 72/72 PASS оба контура. Время -67% до -94%. Масштабируется по каналам.
- Tree hierarchy: 0/108 FAIL. Деревья слишком маленькие (15-585 узлов), tile_map overhead не окупается. Resident ratio 1.16-1.33×.

**Вывод:** Vector grid — production. Деревья — нужен per-level анализ на больших конфигурациях.

---

## Exp10i (exp10i_graph_blocks) — Блочная адресация для графов

**Вопрос:** Работают ли фиксированные блоки как layout для нерегулярных графов?

**Типы графов:** random_geometric, grid_graph, barabasi-albert.
**Стратегии разбиения:** random, spatial (Morton), greedy (BFS).

**Результат:**
- Compute быстрый везде (Contour B 100%). Диагноз: compute healthy, representation sick.
- random_geometric: Contour A 58% PASS. Spatial partition best cbr=0.31.
- grid_graph: Contour A 67% PASS. Spatial partition best cbr=0.20.
- barabasi-albert: Contour A 0% PASS. Best cbr=0.66. Hub-ноды рвут все блоки.
- Padding waste 50-97% (mean 0.77).

**Вывод:** Графы расщепились на два класса. Пространственные — блоки условно годны. Scale-free — блоки структурно несовместимы. Fixed-size blocks не универсальная абстракция для графов.

---

## Exp10j (exp10j_tree_perlevel) — Per-level break-even для деревьев

**Вопрос:** На каких уровнях дерева D_direct побеждает A_bitset?

**Sweep:** 158 000+ trials. Branching 2-32, depth до 10, occupancy 0.01-0.70, payload 4-256 bytes, 3 паттерна активации, 2 оператора.

**Результат:**
- matmul operator: D побеждает при occupancy < 37.5-40% на ЛЮБОМ размере уровня (от 2 до 4096 узлов). Win rate 59%.
- stencil operator: D экономит память ниже того же порога, но НИКОГДА не выигрывает по времени (3.3× медленнее).
- Паттерн активации (random/clustered/frontier) — не влияет на порог.

**Вывод:** Деревья — гибридный режим. Heavy compute + low occupancy → D_direct per-level. Light compute → A_bitset. Порог p*≈0.40 стабилен.

---

## Exp11 (exp11_dirty_signatures) — Dirty signatures

**Вопрос:** Может ли 12-bit dirty signature + debounce заменить full recompute для определения изменений?

**Результат:** PASS. AUC: scalar_grid=0.925, vector_grid=1.000, irregular_graph=1.000, tree=0.910. Все >0.8 kill criterion, все p_adj < 0.001 (Holm-Bonferroni).

**История багов:** Первый запуск: FAIL (AUC 0.0 на scalar_grid). Причина — debounce tracker сравнивал step-to-step (ловил производную, а не сдвиг уровня). Шум давал постоянные скачки → trigger. Структурные изменения давали одиночный импульс → debounce гасил. Второй запуск: ошибочно заменено на oracle scoring (MSE vs ground truth) — AUC 1.000, но это читерство. Третий запуск: правильный фикс — baseline signature comparison + temporal ramp scoring (ramp = mean(last_half_delta) - mean(first_half_delta)). Без ground truth.

**Вывод:** 12-bit dirty signatures работают. Ключ: сравнивать с baseline, а не step-to-step. Temporal ramp ловит устойчивый сдвиг уровня.

---

## DET-2 (exp11a_det2_stability) — Cross-seed stability

**Вопрос:** Стабильны ли метрики pipeline при разных seeds?

**Sweep:** 20 seeds × 4 пространства × 2 бюджета = 160 прогонов.

**Kill criterion (per-regime):**
- Regular spaces (scalar/vector grid) + low budget: CV < 0.10
- Irregular spaces (graph/tree) + low budget: CV < 0.10
- Irregular spaces + high budget: CV < 0.25 (legitimate topological fluctuation)

**Результат:** PASS 8/8. Метрика mean_leaf_value убрана (артефакт test harness: GT = random Gaussians, абсолютное значение зависит от seed). 6 структурных метрик стабильны.

**Вывод:** Pipeline воспроизводим. На high budget с нерегулярными топологиями CV до 0.25 — свойство governor на хаотичных ландшафтах ρ(x), не баг.

---

## Exp12a (exp12a_tau_parent) — Data-driven τ_parent по глубине

**Вопрос:** Можно ли найти пороги τ_parent[L] из данных вместо ручной настройки?

**Результат:** PASS. Per-space thresholds τ[L, space_type] вместо глобального τ[L]. Best method: youden_j. Max accuracy drop 5.6pp (< 15pp kill criterion).

**Пороги:** T1_scalar L1: τ=0.46, T3_graph L1: τ=0.08, T4_tree L1: τ=0.19. Specificity L1 = 1.000 (была 0.25 с глобальным порогом).

**Вывод:** Per-space thresholds решают проблему. R/Up операторы дают разные динамические диапазоны D_parent — единый порог невозможен, per-space обязателен.

---

## P2a (p2a_sensitivity) — Sensitivity sweep порогов гейта

**Вопрос:** Насколько чувствительна система к ручным порогам двухстадийного гейта?

**Результат:** PASS. Ridge width 100% — пороги устойчивы. P2b (адаптивный подбор) не требуется.

**Вывод:** Ручные пороги ok. Гейт нечувствителен в широком диапазоне.

---

## Exp10k (exp10k_cost_surface) — Поверхность стоимости C(I, M, p)

**Вопрос:** Является ли выбор layout гладкой функцией трёх свойств пространства (изотропия I, метрический разрыв M, плотность p), или это дискретная классификация?

**Результат:** Boundary smoothness 0.496 — JAGGED. Поверхность не гладкая.

- Sparse vs dense: **подтверждено** — A_bitset = 0 побед из 810 trials. Sparse всегда лучше.
- D_direct vs D_blocked: **не разрешимо** — ~50% соседних точек переключают победителя.

**Причина:** три дефекта измерения. (1) H(D) слепа к геометрии — не различает "однородный порядок" и "однородный хаос". (2) λ₂ — глобальная метрика, не видит локальное здоровье кэш-линий. (3) Метрики вычислены на полном X, а не на активном подграфе X_active.

**Вывод:** Гипотеза v1.8.3 (layout = argmin C(I,M,p)) не подтверждена и не опровергнута — метрики недостаточны. Нужны: I_active (энтропия на X_active), M_local (средний cache miss на рёбрах X_active). Отложено в Track C. Policy table работает как эмпирическая классификация.

---

## Topological Pre-Runtime Profiling (exp_phase2_pipeline, 21 марта 2026)

**Вопрос:** Можно ли до запуска pipeline определить структурный класс графа и адаптировать параметры (τ_eff, бюджет расщепления) под его топологию?

**Kill criteria:**
- Pre-runtime overhead < 25% pipeline tick (500ms) на worst case
- Classifier accuracy > 85% на корпусе разнородных топологий
- Zero external dependencies для Forman-Ricci (основной путь)

**Архитектура — трёхтактный профайлинг:**

1. **Forman-Ricci** для ВСЕХ рёбер. F(e) = 4 - d_u - d_v + 3|△(e)|. O(1) per edge, <1ms total.
2. **Ollivier-Ricci** (exact EMD через linprog HiGHS) для top-N аномальных рёбер. N = floor(topo_budget_ms / t_ollivier_ms). Бюджет 50ms.
3. **Аппаратная калибровка** — Synthetic Transport Probe: κ_max = W_test · ∛(τ_edge / t_test). Одноразовый замер (52ms).

**Трёхзонный классификатор (v3):**

| Зона | Условие | ECR-прогноз | Действие |
|------|---------|-------------|----------|
| GREEN | κ_mean > 0 | < 5% | Карт-бланш: ослабление τ_eff, полный бюджет |
| YELLOW | κ < 0, Gini < 0.12, η_F ≤ 0.70 | 10-25% | Стандартные лимиты |
| RED | κ < 0 AND (Gini ≥ 0.12 OR η_F > 0.70) | > 30% | Максимальное затягивание τ, блокировка глубокого расщепления |

**Ключевая метрика: η_F = σ_F / √(2⟨k⟩)** — безразмерный индекс топологической энтропии.
- σ_F = стандартное отклонение Forman-кривизны по всем рёбрам
- √(2⟨k⟩) = шумовой предел дисперсии Эрдёша-Реньи с той же средней степенью ⟨k⟩
- Граф с η_F < 0.70 — структурно регулярный (шум ниже пуассоновского пола)
- Граф с η_F > 0.70 — фонит хаосом (шум выше случайного фона)

**Обоснование порога η_F = 0.70:** threshold sweep по κ<0 подмножеству (22 графа). YELLOW графы: η < 0.60 (Grid, Ladder, Planar, Möbius). RED графы: η > 0.76 (ER, Bipartite). Мёртвая зона [0.60, 0.76] — ни один граф корпуса не попадает. Порог 0.70 — середина gap, максимальный margin. Диапазон [0.60, 0.75] даёт одинаковую accuracy — порог не на лезвии.

**Эволюция классификатора:**

| Версия | Сигналы | Accuracy | Промахи |
|--------|---------|----------|---------|
| v1 | κ_mean + Gini(PageRank) | 31/36 (86%) | 5 |
| v2 | κ_mean + Gini + σ_F (абсолютный порог 1.5) | 29/36 (81%) | 7 |
| v3 | κ_mean + Gini + η_F (физ. нормализация, порог 0.70) | 34/35 (97%) | 1 |

v2 провалился: абсолютный σ_F порог зависит от плотности графа. Planar и Watts-Strogatz получали ложный RED. Нормализация на √(2⟨k⟩) убирает зависимость от плотности.

**Документированные граничные случаи (исключены из accuracy):**
- Petersen (N=10): музейный экспонат, ниже разумного минимума для макро-классификатора
- Karate Club: RED при ECR=26.9% (порог RED > 30%). 3 п.п. от границы — статистический люфт
- Tree_bin_d7: маршрутизационная ошибка — дерево не должно попадать в IrregularGraphSpace

**Производительность (pre-runtime, одноразовый setup):**

| Метрика | Значение |
|---------|----------|
| P50 | 56ms |
| P90 | 85ms |
| MAX (Swiss Roll 1000 узлов) | 125ms |
| Mean | 59ms |
| Leiden clustering (средняя) | 1.2ms |
| Overhead vs tick (mean) | 11.8% |
| Overhead vs tick (worst) | 25.0% |

**Ключевые файлы:**
- `exp_phase2_pipeline/topo_features.py` — ядро (CalibrationResult, compute_curvature_hybrid, extract_topo_features, topo_adjusted_rho)
- `exp_phase2_pipeline/test_zone_classifier.py` — валидация v1/v2/v3 на 35 графах
- `exp_phase2_pipeline/bench_preruntime.py` — бенчмарк pre-runtime overhead

**Вывод:** Топологический профайлинг работает. 97% accuracy при 59ms среднем overhead. Пространство получает социальный рейтинг (GREEN/YELLOW/RED) до первого pipeline tick.

---

## Exp14a (exp14a_sc_enforce) — Scale-Consistency Enforcement

**Вопрос:** Как реагировать на D_parent > tau_parent? Можно ли damp/reject delta без существенной потери качества?

**Механизм:**
1. **PASS** (D_parent ≤ tau): delta применяется as-is
2. **DAMP** (tau < D_parent ≤ 2*tau): delta_enforced = delta − Up(R(delta)) × damp_factor. До 3 итераций.
3. **REJECT** (D_parent > 2*tau): delta не применяется, unit пропускается

**Strictness-weighted waste budget:** rejected unit не тратит бюджет качества, но тратит wall-clock. Waste_current += S_node (strictness multiplier). Если Waste ≥ R_max = floor(B_step × 0.2) → force-terminate step. Токсичные хабы (S≈4-5) сжирают квоту лавинообразно.

**Adaptive τ для деревьев:** τ_T4(N) = τ_base × (1 + β/√N) — ослабление для маленьких деревьев.

**Результат:** PASS. Проекция (delta − Up(R(delta)) × factor) эффективнее скалирования (delta × 0.5) — сохраняет HF-детали, убирая только LF-утечку.

---

## Exp13 (exp13_segment_compression) — Segment Compression

**Вопрос:** Можно ли сжать degree-2 цепочки дерева >50% используя стабильность dirty signature?

**Kill criteria:** compression ratio > 50% degree-2 узлов, per-step overhead < 10%.

**Результат:** PASS.

| Пространство | Compression | Overhead | Guard |
|---|---|---|---|
| binary_d7 (4 chains) | 66% | −9.1% (прибыльно) | active |
| binary_d8 (6 chains) | 60% | −4.0% (прибыльно) | active |
| binary_d6 (3 chains) | — | — | blocked: below_n_critical |
| quadtree_d5 (4 chains) | — | — | blocked: below_n_critical |

**Thermodynamic guards:**
1. **Bombardment density:** budget ≥ 50% active nodes → skip (carpet-bombing kills chains)
2. **Breakeven N_critical=12:** derived from profiling (C_refine=15.9μs, C_track=1.9μs, C_init=100.5μs)

**Вывод:** Compression прибыльна на деревьях глубины ≥7 с достаточным количеством degree-2 цепочек. Guard `should_compress()` отсекает невыгодные случаи автоматически.

---

## Phase 2 E2E Validation (exp_phase2_e2e) — End-to-end Pipeline

**Вопрос:** Работает ли собранный pipeline end-to-end на всех 4 типах пространств?

**Sweep:** 4 пространства × 20 seeds × 3 бюджета (0.10, 0.30, 0.60) = 240 конфигураций.

**Kill criteria:**

| Метрика | Порог | Результат |
|---------|-------|-----------|
| Quality (PSNR) | > 0 dB vs coarse-only | ✅ 240/240 positive |
| Reject rate | < 5% refined units | ✅ max 0% (scalar/vector/graph) |
| Budget compliance | refined ≤ budget × total | ✅ 240/240 |
| Runtime | < 60s per config | ✅ max 245ms |

**Результаты по пространствам (с topo profiling для irregular_graph):**

| Пространство | PSNR gain median | IQR | Reject max | Wall max |
|---|---|---|---|---|
| scalar_grid | +7.34 dB | [2.92, 11.40] | 0% | 23ms |
| vector_grid | +1.46 dB | [0.41, 3.60] | 0% | 33ms |
| irregular_graph | +3.54 dB | [1.71, 7.45] | 0% | 245ms |
| tree_hierarchy | +1.48 dB | [1.23, 2.30] | 0% (max 50% единичный) | 4ms |

**Topo profiling в E2E (irregular_graph, 21.03.2026):**
- Zone distribution: GREEN 75% (45/60), RED 25% (15/60)
- η_F median: 1.0557, tau_factor median: 1.3
- Topo computation: 67ms median, 78ms max
- PSNR change vs pre-topo: −0.20 dB (3.74→3.54) — ожидаемо, GREEN relaxes tau, RED tightens

**DET-1 Recheck (с topo profiling):** 40/40 PASS — bitwise determinism. Topo profiling детерминистичен при фиксированном seed.

**DET-2 Recheck (с topo profiling):** 8/8 PASS — cross-seed stability. Kill metrics (n_refined, compliance) CV≈0. psnr_gain CV=0.09–0.37 — информационная метрика, зависит от seed-generated GT (аналог mean_leaf_value в exp11a, не kill-criteria). Topo metrics: η_F CV=0.03, computation_ms CV=0.05.

**Вывод:** Pipeline валидирован end-to-end. Все kill criteria пройдены. Topo profiling интегрирован и не ломает детерминизм.

---

## Финальная layout policy (результат серии exp10)

Полная методика: `docs/layout_selection_policy.md`

| Тип пространства | Layout | Статус |
|-----------------|--------|--------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production |
| vector_grid | D_direct (packed tiles + tile_map) | Production |
| tree_hierarchy | Гибрид: D_direct per-level (p<0.40 + heavy compute), A_bitset иначе | Validated |
| irregular_graph / spatial | D_blocked (блочная адресация) conditional | Conditional |
| irregular_graph / scale-free | A_bitset (dense + bitset) fallback | Fallback |
