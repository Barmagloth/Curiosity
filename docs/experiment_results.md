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

**Результат:** FAIL 3/4 пространств. AUC 0.0-0.37 на scalar_grid, vector_grid, tree. Только irregular_graph AUC 0.99.

**Вывод:** Архитектурная проблема. Signature слишком грубая для детектирования мелких изменений на регулярных пространствах. Требует переработки.

---

## Exp12a (exp12a_tau_parent) — Data-driven τ_parent по глубине

**Вопрос:** Можно ли найти пороги τ_parent[L] из данных вместо ручной настройки?

**Результат:** Thresholds найдены. τ убывает с глубиной как предсказано. Но L1 specificity низкая.

**Вывод:** Частичный успех. Пороги существуют и согласуются с теорией. L1 enforcement требует доработки.

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

## Финальная layout policy (результат серии exp10)

Полная методика: `docs/layout_selection_policy.md`

| Тип пространства | Layout | Статус |
|-----------------|--------|--------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production |
| vector_grid | D_direct (packed tiles + tile_map) | Production |
| tree_hierarchy | Гибрид: D_direct per-level (p<0.40 + heavy compute), A_bitset иначе | Validated |
| irregular_graph / spatial | D_blocked (блочная адресация) conditional | Conditional |
| irregular_graph / scale-free | A_bitset (dense + bitset) fallback | Fallback |
