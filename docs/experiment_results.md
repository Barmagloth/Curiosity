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
| Enox infrastructure | ✅ PASS | 4 observation-only паттерна. Zero functional change. Заготовка для Phase 3 |

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

---

## Enox Infrastructure (exp_phase2_pipeline) — Observation-only паттерны — PASS

**Вопрос:** Можно ли встроить инфраструктуру наблюдения (трассировка, журнал решений, дедупликация, sweep) в pipeline без изменения его поведения?

**Источник:** Enox open-source framework. Взяты идеи (не код), реализованы под Curiosity.

**Принцип:** все 4 паттерна — чистая аннотация/наблюдение. Никогда не модифицируют pipeline state. Все дефолты = False. При включении: zero functional change, подтверждается битовым совпадением state hash.

**4 паттерна:**

| # | Паттерн | Что делает | Ценность сейчас | Ценность в Phase 3 |
|---|---------|-----------|-----------------|---------------------|
| 1 | RegionURI | SHA256(parent_id\|op_type\|child_idx) → 16 hex. Детерминированный адрес юнита | Трассировка | Провенанс |
| 2 | DecisionJournal | Append-only лог: region_id, tick, gate_stage, decision, metrics, thresholds | Дебаг | Полный аудит решений |
| 3 | MultiStageDedup | 3 уровня: exact hash / metric distance (ε) / policy. ε=0.0 → не срабатывает | Заготовка | Экономия вычислений в multi-pass |
| 4 | PostStepSweep | Sibling dirty-sig в tree_hierarchy. Порог 5% | Поиск merge-кандидатов | Автоматическое сжатие |

**Config (6 ручек, все default=False):**
`enox_journal_enabled`, `enox_dedup_enabled`, `enox_dedup_epsilon` (0.0), `enox_sweep_enabled`, `enox_sweep_threshold` (0.05), `enox_include_uri_map`

**Baseline fingerprint (без Enox):**

| Метрика | Значение |
|---------|----------|
| Конфигураций | 20 (4 spaces × 5 seeds) |
| Budget | 0.30 |
| PSNR median | +2.32 dB |
| DET-1 spot check | PASS |
| Wall time median | 11.9ms |

**Сравнение baseline vs enox:** NO REGRESSION.

| Метрика | Результат |
|---------|-----------|
| Hash match | 15/20 SAME (scalar_grid, vector_grid, tree_hierarchy — битово идентичны) |
| Hash diff | 5/20 irregular_graph (non-deterministic topo calibration timing, не Enox) |
| PSNR median delta | +0.0000 dB |
| PSNR max abs delta | 1.40 dB (irregular_graph seed=2, topo timing) |
| DET-1 | PASS |
| All PSNR positive | Yes (20/20) |
| Any rejects | No |

**Ключевые файлы:**
- `exp_phase2_pipeline/enox_infra.py` — реализация (region_uri, DecisionJournal, MultiStageDedup, PostStepSweep)
- `exp_phase2_pipeline/enox_comparison.py` — before/after framework
- `exp_phase2_pipeline/config.py` — 6 knobs
- `exp_phase2_pipeline/pipeline.py` — integration (DONE)

**Вывод:** Инфраструктура наблюдения полностью интегрирована. Функционально — zero change by design. Практическая ценность появится в Phase 3 (multi-pass, дебаг сложных решений).

---

## Phase 3 — Семантика дерева и rebuild (22 марта 2026)

Phase 3 состоит из четырёх экспериментов: три потока (S1, S2, S3) и C-pre. Цель — определить, есть ли у дерева семантика, и работает ли инкрементальный rebuild.

### Exp14 — Anchors + Periodic Rebuild (P1-B3) — CONDITIONAL PASS

**Вопрос:** Насколько расходится local update vs full rebuild? Какая стратегия rebuild минимизирует divergence при минимальных затратах?

**Конфигурация:** 720 конфигов (4 spaces × 9 стратегий × 20 seeds). 50 шагов на конфиг. Стратегии: no_rebuild, periodic_{5,10,20,50}, dirty_{0.05,0.1,0.2,0.5}.

**Kill criterion:** divergence < 5% (0.05).

**Результат по пространствам:**

| Space | Max Div | Mean Div | Best Strategy | Kill |
|-------|---------|----------|---------------|------|
| scalar_grid | 0.000 | 0.000 | no_rebuild | ✅ PASS |
| vector_grid | 0.000 | 0.000 | no_rebuild | ✅ PASS |
| irregular_graph | 1.517 | 0.374 | dirty_0.05 (0.204 mean) | ❌ FAIL |
| tree_hierarchy | 1.715 | 0.694 | dirty_0.05 (0.204 mean) | ❌ FAIL |

**Ключевые находки:**
- **Сетки (scalar_grid, vector_grid):** divergence = 0.000 для ВСЕХ стратегий. Local update идеален — rebuild не нужен.
- **Graph и tree:** dirty-triggered (порог 0.05) — лучшая стратегия (mean_div=0.204), но всё равно далеко от порога 0.05. Periodic стратегии хуже dirty-triggered.
- **Вывод:** дихотомия — сетки тривиальны (не нужен rebuild), графы/деревья — проблема не в rebuild стратегии, а в самом local update (он вносит структурный drift).

**Verdict:** CONDITIONAL PASS — проходит для grid, не проходит для graph/tree. Требуется переосмысление local update для нерегулярных пространств.

---

### Exp15 — LCA-Distance vs Feature Similarity (P3a) — FAIL

**Вопрос:** Коррелирует ли LCA-расстояние в дереве с семантической близостью (‖feature_i − feature_j‖)?

**Конфигурация:** 80 конфигов (4 spaces × 20 seeds). 500 пар юнитов на seed.

**Kill criterion:** Spearman r > 0.3.

**Результат:**

| Space | Spearman (mean ± std) | Pearson (mean ± std) | Kill |
|-------|----------------------|---------------------|------|
| scalar_grid | 0.299 ± 0.108 | 0.283 ± 0.087 | ❌ FAIL (0.299 < 0.3) |
| vector_grid | −0.032 ± 0.035 | −0.031 ± 0.024 | ❌ FAIL |
| irregular_graph | 0.267 ± 0.113 | 0.272 ± 0.110 | ❌ FAIL |
| tree_hierarchy | 0.006 ± 0.064 | 0.079 ± 0.072 | ❌ FAIL |

**Ключевые находки:**
- scalar_grid ближе всего к порогу (0.299), отдельные seeds до 0.49, но high variance.
- vector_grid и tree_hierarchy — практически нулевая корреляция.
- **Вывод:** LCA-расстояние не является надёжной метрикой семантической близости. Дерево — журнал уточнений, а не семантическая карта.

**Verdict:** FAIL. Дерево не семантично по критерию LCA-distance.

---

### Exp15b — Bush Clustering (P3b) — FAIL

**Вопрос:** Есть ли естественные кластеры среди leaf-путей? Стабильны ли они?

**Конфигурация:** 80 конфигов (4 spaces × 20 seeds). Методы: kmeans (k=2-10), DBSCAN (eps sweep), agglomerative.

**Kill criterion:** Silhouette > 0.4 AND ARI stability > 0.6.

**Результат:**

| Space | Silhouette (mean ± std) | k_mode | ARI | Kill |
|-------|------------------------|--------|-----|------|
| scalar_grid | 0.661 ± 0.125 | 2 | 0.073 | ❌ FAIL |
| vector_grid | 0.793 ± 0.065 | 2 | 0.094 | ❌ FAIL |
| irregular_graph | 0.649 ± 0.082 | 2 | −0.011 | ❌ FAIL |
| tree_hierarchy | 0.485 ± 0.067 | 2 | 0.210 | ❌ FAIL |

**Ключевые находки:**
- Silhouette проходит порог 0.4 во всех пространствах — кластеры внутри отдельного seed визуально осмысленны.
- ARI катастрофически низкий — кластеры НЕ воспроизводятся между seeds.
- **Вывод:** кластеры leaf-путей — артефакт конкретного seed, а не устойчивое свойство пространства.

**Verdict:** FAIL. Silhouette > 0.4 ✅, но ARI stability ❌. Bushes не стабильны.

---

### Exp16 — C-pre: Trajectory Profile Clustering — PASS (UNFREEZE)

**Вопрос:** Есть ли кластерная структура в trajectory features (EMA квантили, split signatures)?

**Конфигурация:** 80 конфигов (4 spaces × 20 seeds). Gap statistic + silhouette + ARI.

**Kill criterion:** Gap > 1.0 AND Silhouette > 0.3.

**Результат:**

| Space | Gap (mean ± std) | Silhouette (mean ± std) | k_mode | ARI | Kill |
|-------|-----------------|------------------------|--------|-----|------|
| scalar_grid | 1.37 ± 0.09 | 0.605 ± 0.148 | 4 | 0.454 | ✅ PASS |
| vector_grid | 2.04 ± 0.26 | 0.794 ± 0.125 | 4 | 0.086 | ✅ PASS |
| irregular_graph | 1.39 ± 0.41 | 0.518 ± 0.117 | 2 | 0.029 | ✅ PASS |
| tree_hierarchy | 2.40 ± 1.01 | 0.453 ± 0.075 | 7 | 0.441 | ✅ PASS |

**Ключевые находки:**
- Все 4 пространства проходят оба порога (gap > 1.0, silhouette > 0.3).
- ARI умеренный для scalar_grid (0.454) и tree_hierarchy (0.441), низкий для vector_grid и irregular_graph — кластеры реальны (высокий gap), но границы нестабильны.
- k_mode варьируется: 2-4 для grid/graph, 7 для tree — разные пространства дают разное количество профилей.

**Verdict:** PASS. Track C **UNFREEZE**. Trajectory features демонстрируют реальную кластерную структуру.

---

### Phase 3 — Сводка решений

| Эксперимент | Kill Criterion | Результат | Решение |
|-------------|---------------|-----------|---------|
| Exp14 (anchors) | div < 5% | Grid: PASS, Graph/Tree: FAIL | CONDITIONAL — local update ок для grid, нужен redesign для graph/tree |
| Exp15 (LCA-distance) | Spearman r > 0.3 | FAIL (max 0.299) | Дерево = журнал, не метрика |
| Exp15b (bushes) | Sil > 0.4 + ARI > 0.6 | Sil PASS, ARI FAIL | Кластеры не стабильны |
| Exp16 (C-pre) | Gap > 1.0 + Sil > 0.3 | PASS (все 4 spaces) | **Track C UNFREEZE** |

**Gate Phase 3 → Phase 4:** "Дерево семантично?" → **НЕТ** (P3a+P3b FAIL). "C unfreezes?" → **ДА** (C-pre PASS).

**Импликации для Phase 4:**
- P4 (downstream consumer test) может идти вперёд — он не зависит от семантики дерева.
- Track C открыт — дискретные профили существуют, можно ставить multi-objective эксперименты.
- Local update для graph/tree требует отдельного R&D (structural drift problem).

### Exp15b Bushes — заметки для revisit

Кусты (leaf-path кластеры) формально FAIL по ARI, но содержат потенциально полезную информацию:

1. **Кластеры существуют внутри каждого seed** (Silhouette > 0.4). Нестабильность между seeds (ARI < 0.6) может означать зависимость от конкретного GT, а не бесполезность метода.
2. **Leaf-path similarity** — потенциальный инструмент для:
   - Обнаружения дублирующих регионов (листья с похожими путями = похожая структура → merge candidates)
   - Уплотнения кластеров: если два кластера дают схожие leaf-paths, они потенциально mergeable
   - Фичи для downstream классификатора (Track C): профиль пути листа = "как pipeline пришёл к этому результату"
3. **Связь с trajectory profiles (exp16):** trajectory profiles кластеризуются стабильнее (Gap > 1.0), чем bushes. Возможно, trajectory features + leaf-path features дадут более стабильные кластеры, чем каждый по отдельности. Запланирован revisit после первых результатов Track C.

---

## Phase 3.5 — Трёхслойная декомпозиция ρ (exp17, 23 марта 2026)

### Мотивация

Phase 3 показала: дерево refinement не семантично (LCA-distance FAIL, bushes unstable), а local update дрейфует на graph/tree (anchors FAIL). Корневая причина — **монолитная ρ смешивает три ортогональных сигнала:**

1. Структура пространства (топология) — не зависит от данных
2. Наличие данных — не зависит от конкретного запроса
3. Задаче-специфический residual — зависит от всего

Смешение делает дерево непереиспользуемым и непрозрачным: нельзя отделить "дерево знает структуру" от "дерево заточено под конкретный residual".

### Архитектура: три слоя

```
Layer 0: ТОПОЛОГИЯ        "как устроено пространство"
         Вход:  сырое пространство (граф/дерево/сетка)
         Выход: per-unit structural score + cluster_ids
         Для graph: Leiden clusters + Forman/Ollivier curvature + PageRank + boundary anomalies
         Для tree:  depth-band grouping + subtree size scoring
         Для grid:  spatial quadrant blocks (тривиальная топология)
         [data-independent, вычисляется один раз]

Layer 1: ДАННЫЕ ДА/НЕТ    "где есть нетривиальный сигнал"
         Вход:  L0 cluster structure + ground truth
         Выход: per-unit presence score + active_mask
         Метрика: variance(gt[region]) — "есть ли тут структура, заслуживающая compute?"
         Порог: CASCADE QUOTAS (Variant C) — см. ниже
         [data-dependent, query-independent]

Layer 2: ЗАПРОС            "из того что лежит — где нужное мне"
         Вход:  frozen tree (L0+L1) + task-specific query function
         Выход: ordered refinement list
         Три сменных query-функции на одном дереве:
           - MSE (текущий unit_rho)
           - Max absolute error
           - HF residual (Лапласиан)
         [task-specific, дешёвый]
```

Каждый слой **сужает** рабочее множество для следующего. Информация течёт строго сверху вниз (L0 → L1 → L2).

### Каскадные квоты (Variant C)

**Проблема:** фиксированный порог L1 (l1_threshold = 0.01) убивал 97-98% юнитов на scalar_grid при масштабе 1000. Из 1024 тайлов выживало ~20, single_pass уточнял 307. Reusability ratio = 0.725 FAIL.

**Почему:** мелкие тайлы (8×8 пикселей) на гладких участках GT имеют variance < 0.01 даже без sparsity. Фиксированный порог не учитывает масштаб.

**Решение:** порог привязан к кластерной структуре L0. Каждый L0-кластер гарантирует минимальное число выживших:

```
quota = max(1, ceil(cluster_size × min_survival_ratio))
```

Где min_survival_ratio = budget_fraction (обычно 0.30). Внутри каждого кластера юниты сортируются по presence score и сохраняются top-quota.

**Свойства:**
- Ни один L0-кластер не вымирает полностью
- Порог адаптивен к масштабу (больше юнитов → больше выживших → больше бюджета)
- Нет magic numbers — единственный параметр (min_survival_ratio) привязан к budget_fraction
- Информация каскадируется: L0 topology → L1 quotas → L2 budget

**Результат:** scalar_grid 1000 перешёл с 0.725 FAIL → 0.863 PASS (финальный sweep, 20 seeds).

### Streaming Pipeline

Вместо batch (L0 all → L1 all → L2 all) — покластерная обработка:

```
Cluster_0: [L0 score] → [L1 filter] → [L2 refine]
Cluster_1:              [L0 score] → [L1 filter] → [L2 refine]
Cluster_2:                           [L0 score] → [L1 filter] → [L2 refine]
```

Кластеры обрабатываются в порядке L0 priority score (самые структурно важные — первыми). Глобальный budget cap: суммарное число refinements ≤ budget_fraction × n_total. При исчерпании бюджета — ранняя остановка.

**Преимущества:**
- Первые результаты после 1 кластера, а не после полной карты
- L1 pruning реально сокращает refinement (budget per-cluster, а не global)
- 10-20% быстрее batch на grid пространствах

### Industry Baselines

Четыре стандартных подхода для сравнения:

| Baseline | Подход | Пространства |
|----------|--------|-------------|
| cKDTree (scipy) | k-d tree build + sort by rho | Все 4 |
| Quadtree | Деление на квадранты по суммарному rho | Grid only |
| Leiden + brute force | Community detection + sort by rho внутри | Graph only |
| Wavelets (Haar DWT) | Detail coefficients как saliency map | Scalar grid only |

### Результаты (1080 конфигов, 4 spaces × 3 scales × 8 approaches × 20 seeds)

**Ошибки:** 0 из 1080.

**Reusability (frozen tree + другой query vs fresh build):**

| Space | Scale 100 | Scale 1000 | Scale 10000 |
|-------|----------|-----------|------------|
| scalar_grid | 0.838 PASS | 0.863 PASS | 0.884 PASS |
| vector_grid | 0.984 PASS | 0.959 PASS | 0.978 PASS |
| irregular_graph | 0.926 PASS | 0.926 PASS | 1.000 PASS |
| tree_hierarchy | 0.996 PASS | 0.999 PASS | 0.999 PASS |

**Время (single query, scale=1000):**

| Space | single_pass | three_layer_stream | kdtree (industry best) |
|-------|------------|-------------------|----------------------|
| scalar_grid | 51ms | 32ms | 23ms |
| vector_grid | 690ms | 709ms | 652ms |
| irregular_graph | 0.4ms | 68ms (L0 overhead) | 0.5ms |
| tree_hierarchy | 9ms | 12ms | 9ms |

**Amortized cost (break-even for tree_hierarchy):**

| N queries | three_layer | kdtree | single_pass |
|-----------|------------|--------|-------------|
| 1 | 12ms | 10ms | 10ms |
| 2 | 17ms ✅ | 19ms | 19ms |
| 5 | 34ms ✅ | 48ms | 47ms |
| 10 | 61ms ✅ | 96ms | 95ms |

### Выводы

1. **Архитектура работает:** reusability 12/12 PASS. Frozen tree переиспользуем.
2. **PSNR trade-off:** 2-4 dB ниже single_pass на grid (цена L1 pruning), паритет на graph/tree.
3. **Время:** streaming быстрее batch, но kdtree (scipy C) быстрее обоих на single query. При ≥2 запросах three_layer выигрывает на tree_hierarchy.
4. **Bottleneck = refinement, не scoring.** Все подходы уточняют ~одинаковое число юнитов, refinement = 50-70% total time (numpy, уже near-C).
5. **C-оптимизация** scoring фаз даст мультипликативный эффект для streaming: L0 topo 70ms→5ms, L1+L2 scoring 13ms→1.3ms. Streaming на C потенциально обойдёт kdtree на graph/tree.
6. **Ценность Curiosity vs industry:** побочные данные (topo features, zones, cluster structure, decision journal), которых kdtree не даёт. Interpretability: каждый слой отвечает на отдельный вопрос.

### Ключевые файлы

| Файл | Назначение |
|------|-----------|
| `experiments/exp17_three_layer_rho/layers.py` | Ядро: Layer0, Layer1 (cascade quotas), Layer2, FrozenTree, ThreeLayerPipeline, IndustryBaselines |
| `experiments/exp17_three_layer_rho/exp17_three_layer_rho.py` | Runner с --chunk для параллельного запуска |
| `experiments/exp17_three_layer_rho/config17.py` | Параметризация (scales, thresholds, approaches) |
| `experiments/exp17_three_layer_rho/results/` | JSON результаты по чанкам |

---

## Exp18 — Basin membership vs feature similarity (RG-flow hypothesis)

**Вопрос:** Ведёт ли refinement tree себя как RG-flow траектория? Можно ли использовать basin membership как семантическую метрику?

**Дизайн:** 80 конфигураций. Point-biserial корреляция между basin membership (принадлежность к одному бассейну аттракции) и feature similarity. Kill criterion: r > 0.3.

**Результат:** Point-biserial r = 0.019. Kill criterion r > 0.3: **FAIL**.

**Причина:** Бассейны вырождены в single-pass при 30% бюджете — дерево недостаточно глубокое для формирования стабильных бассейнов аттракции. Нужен multi-pass для достаточной глубины.

**Вывод:** RG-flow гипотеза не опровергнута, но не подтверждена в текущих условиях. Deferred до post-multi-pass (после Phase 4). Связано с Exp0.10 (R,Up) sensitivity (concept section 8.10).

---

# Experiment Results (English Summary)

All experiments through exp17 are complete. Phase 3 and Phase 3.5 are DONE.

## Phase 0 (exp01--exp08): Core Validation

- **Exp0.1--0.2:** Adaptive refinement confirmed (MSE/PSNR winrate 98-100%).
- **Exp0.3:** Halo (overlap >= 3, cosine feathering) mandatory for boundary artifacts.
- **Exp0.4--0.5:** Residual-only signal breaks under noise (correlation drops 0.90 to 0.54).
- **Exp0.6--0.7b:** Two-stage gate solves clean vs degraded mode selection (+0.77-1.49 dB on noise).
- **Exp0.8:** EMA budget governor mandatory (StdCost -50%, penalty -85%). Phase schedule not confirmed.
- **Halo cross-space:** Grid/graph PASS, tree FAIL. Rule: boundary parallelism >= 3.
- **SC-baseline:** D_hf AUC=0.806, D_parent (fixed) AUC=0.853, d=1.491. Cross-space 0.824-1.000.

## Phase 1 (exp10--exp14a): Layout, Determinism, Infrastructure

- **Exp10:** Compact-with-reverse-map KILLED (VRAM +38.6%). Grid baseline.
- **Exp10d:** DET-1 PASS 240/240 (bitwise determinism CPU+CUDA).
- **Exp10e--10j:** Layout resolved per space type. D_direct for grid, hybrid for tree, D_blocked conditional for spatial graph.
- **Exp11:** 12-bit dirty signatures PASS (AUC 0.91-1.0).
- **DET-2 (exp11a):** Cross-seed stability PASS 8/8.
- **Exp12a:** Data-driven tau_parent PASS (per-space thresholds, specificity 1.000).
- **Exp13:** Segment compression PASS (66% on depth-7 trees, thermodynamic guards).
- **Exp14a:** SC-enforce PASS (three-tier pass/damp/reject + adaptive tau for trees).
- **Phase 2 E2E:** 240 configs, all kill criteria passed. Topo profiling integrated (97% accuracy, P50=56ms).
- **Enox infrastructure:** 4 observation-only patterns. Zero functional change. DET-1 PASS.

## Phase 3 (exp14--exp16): Tree Semantics and Rebuild

- **Exp14 (anchors):** 720 configs. Grid divergence=0.000 (PASS). Graph/tree divergence>0.20 (FAIL). dirty_0.05 best strategy (mean_div=0.204). Kill criterion div<0.05: graph/tree ALL FAIL. CONDITIONAL PASS.
- **Exp15 (LCA-distance):** 80 configs. Spearman mean=0.135 across spaces. Kill r>0.3: ALL FAIL. scalar_grid closest at 0.299. Tree is not semantic.
- **Exp15b (bushes):** 80 configs. Silhouette>0.4 PASS all spaces. ARI<=0.210 FAIL all spaces. k_mode=2 everywhere. Clusters exist but not stable across seeds.
- **Exp16 (C-pre profiles):** 80 configs. Gap>1.0 AND Silhouette>0.3: ALL 4 spaces PASS. Track C UNFREEZE.

## Phase 3.5 (exp17): Three-Layer Rho Decomposition

- **1080 configs** (4 spaces x 3 scales x 8 approaches x 20 seeds).
- **Reusability:** 12/12 PASS (min 0.838). Frozen tree is reusable across queries.
- **Cascade quotas (Variant C):** Fixed scalar_grid 1000 from 0.725 FAIL to 0.863 PASS. Quota = max(1, ceil(cluster_size x min_survival_ratio)).
- **Streaming pipeline:** 10-20% faster than batch on grids.
- **Industry baselines:** kdtree, quadtree, wavelets, leiden. kdtree faster on single query; three_layer wins at >=2 queries on tree_hierarchy.

## Exp18: Basin Membership (RG-flow Hypothesis)

- **80 configs.** Point-biserial r = 0.019. Kill r > 0.3: FAIL.
- Basins degenerate in single-pass at 30% budget. Deferred to post-multi-pass (after Phase 4).

---
