# Curiosity — Иерархия экспериментов (v2.2)

Документ фиксирует актуальный статус, зависимости и порядок экспериментов.

Обновлено после Phase 3.5 (exp17, трёхслойная декомпозиция ρ). Все эксперименты до exp17 включительно завершены.

v2.2: Phase 3 и Phase 3.5 завершены. exp14-exp17 добавлены в маппинг и граф зависимостей.
v2.1: добавлен уровень DET (детерминизм и воспроизводимость) — кросс-инфраструктурное требование.

---

# Маппинг: папки → вопросы → план валидации

| Папка | Вопрос | План валидации | Статус |
|-------|--------|----------------|--------|
| `exp01_poc/` | Adaptive refinement работает? | — | ✅ Да (PoC) |
| `exp02_cifar_poc/` | (то же, CIFAR) | — | ✅ Да |
| `exp03_halo_diagnostic/` | Halo обязателен? | — | ✅ Да |
| `phase1_halo/` | Halo: r_min, blending, hardened | §A (A1+A2+A3) | ✅ Закрыт |
| `exp04_combined_interest/` | Комбинированная интересность нужна? | — | ✅ Да |
| `exp05_break_oracle/` | Oracle-free проверка | — | ✅ Да |
| `exp06_adaptive_switch/` | Авто-переключение ρ | — | ✅ Да |
| `exp07_gate/` | Двухстадийный гейт | — | ✅ Да |
| `exp08_schedule/` | Schedule + governor + probe | — | ✅ Закрыт |
| `phase2_probe_seam/` | Probe + SeamScore validation | §B (B1+B2) | ✅ Закрыт |
| `exp09a_layout_sandbox/` | Layout: grid vs compact (microbench) | §C (C3/Exp0.9a) | ✅ Частично |
| `halo_crossspace/` | Halo applicability across space types | Phase 0 | ✅ Закрыт (правило выведено) |
| `sc_baseline/` | Scale-consistency D_parent/D_hf verification | Phase 0 (SC-0..SC-4) | ✅ Закрыт (SC-5 open) |
| `p2a_sensitivity/` | Sensitivity sweep порогов гейта | P2a | ✅ PASS (ridge=100%, MANUAL_OK) |
| `exp10_buffer_scaling/` | Grid vs compact-with-reverse-map на GPU | P0 (0.9b0) | ✅ KILL compact (VRAM +38.6%). Grid baseline. |
| `exp10d_seed_determinism/` | Побитовый детерминизм при фикс. seed | DET-1 | ✅ PASS (240/240) |
| `exp10e_tile_sparse/` | Tile-sparse layouts без global reverse_map | P0 (0.9b1) | ✅ A alive, B/C killed. See exp10f. |
| `exp10f_packed_lookup/` | Packed tiles + alternative lookup (hash/direct) | P0 (0.9b2) | ⚠️ D passes Contour A, fails Contour B (peak VRAM from conv2d workspace). E archived as contingency. |
| `exp10g_dual_benchmark/` | Dual-mode benchmark: manual stencil (layout cost) vs conv2d (operator cost). Resolves D's Contour B. | P0 (0.9b3) | ✅ D_direct PASS both contours. -54% to -80% time, -36% to -86% peak VRAM. |
| `exp10h_cross_space/` | D_direct on vector_grid and tree_hierarchy | P0 (0.9b3) | ✅ vector_grid 72/72 PASS. tree FAIL 0/108 (configs too small → exp10j). |
| `exp10i_graph_blocks/` | Graph block-based addressing with 3 partition strategies | P0 (0.9b3) | ✅ Spatial graphs conditional (cbr≤0.35). Scale-free rejected (cbr=0.66). |
| `exp10j_tree_perlevel/` | Per-level independent D_direct vs A_bitset benchmark for trees. Finds break-even thresholds per level. | P0 (0.9b3) | ✅ matmul: D wins at p<0.40 any N_l. stencil: D saves memory, never time. Contour B 45% PASS. |
| `exp11_dirty_signatures/` | 12-bit dirty signature + debounce | P1-B2 | ✅ PASS (AUC 0.91-1.0, baseline comparison + temporal ramp) |
| `exp11a_det2_stability/` | Cross-seed stability (DET-2) | DET-2 | ✅ PASS 8/8 (per-regime CV thresholds) |
| `exp12a_tau_parent/` | Data-driven τ_parent[L] per depth | SC-5 | ✅ PASS (per-space thresholds, specificity 1.000) |
| `exp13_segment_compression/` | Segment compression with thermodynamic guards (N_critical=12, bombardment guard) | P1-B1 | ✅ PASS (overhead eliminated on small trees) |
| `exp10k_cost_surface/` | Cost surface C(I, M, p) for layout selection. Boundary smoothness 0.496 — JAGGED. Sparse always beats dense, but D_direct vs D_blocked not separable. Metrics insufficient. | P0 | ⚠️ Inconclusive (deferred to Track C) |
| `exp14a_sc_enforce/` | Scale-consistency enforcement: three-tier pass/damp/reject + strictness-weighted waste budget + adaptive τ T4(N)=τ_base*(1+β/√N) | SC-enforce | ✅ PASS |
| `exp_phase2_pipeline/` | Full pipeline assembly (gate + governor + SC-enforce + probe + traversal) + topological pre-runtime profiling (hybrid Forman/Ollivier curvature, three-zone classifier v3, η_F entropy index) + Enox infrastructure (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep — observation-only, all defaults False) | Phase 2 + Topo + Enox | ✅ PASS |
| `exp_phase2_e2e/` | End-to-end validation: 4 space types, 240 configs. DET-1 recheck 40/40 + DET-2 recheck 8/8 (with topo profiling). irregular_graph re-run with zone classification (GREEN 75%/RED 25%) | Phase 2 | ✅ PASS |
| `exp_deferred_revisit/` | Research note: Morton/block-sparse/schedule | — | ✅ Done |
| `exp14_anchors/` | Anchors + periodic rebuild: local update vs full rebuild divergence. 720 configs (4 spaces × 9 strategies × 20 seeds). Grid: div=0.000 (PASS). Graph/tree: div>0.20 (FAIL). Dirty-triggered > periodic. | P1-B3 | ⚠️ CONDITIONAL (grid PASS, graph/tree FAIL) |
| `exp15_lca_semantics/` | LCA-distance vs feature similarity correlation. 80 configs. Spearman: scalar_grid 0.299, vector_grid −0.032, irregular_graph 0.267, tree_hierarchy 0.006. | P3a | ❌ FAIL (r < 0.3 all spaces) |
| `exp15b_bushes/` | Leaf-path clustering: silhouette + cross-seed ARI stability. 80 configs. Silhouette >0.4 all spaces, but ARI ≤0.210 everywhere. | P3b | ❌ FAIL (ARI unstable) |
| `exp16_cpre_profiles/` | C-pre trajectory profile clustering. 80 configs. Gap >1.0, Silhouette >0.3 all spaces. Track C UNFREEZE signal. | C-pre | ✅ PASS (UNFREEZE) |
| `exp17_three_layer_rho/` | Three-layer rho decomposition. 1080 configs (4 spaces × 3 scales × 8 approaches × 20 seeds). Cascade quotas (Variant C). Streaming pipeline. Reusability 12/12 PASS (min 0.838). Industry benchmarks (kdtree, quadtree, wavelets, leiden). | Phase 3.5 | ✅ PASS (reusability) |
| `exp18_basin_membership/` | Basin membership vs feature similarity. 80 configs. Point-biserial r=0.019, kill r>0.3: FAIL. Basins degenerate in single-pass. Deferred to post-multi-pass. | RG-flow (post-Phase 4) | ❌ FAIL (deferred) |
| `exp19_multi_tick_sweep/` | Multi-tick pipeline validation. 5 sub-experiments (19a-e). 2050 configs. Scaling law, gate stress, param sweep, real data, noisy/hetero. Key: multi-tick +2-7% under noise, neutral on clean. ROI fix (global→local). | Phase 4 | ✅ DONE (2050 configs) |

**Примечание Phase 2:** Graph clustering upgraded from k-means to Leiden (community detection), validated on 10 pathological topologies: Swiss Roll, Barbell, Hub-Spoke, Ring of Cliques, Bipartite, Erdos-Renyi, Grid, Planar Delaunay, Mobius strip.

**Примечание Topo Profiling (21.03.2026):** Topological pre-runtime profiling added to IrregularGraphSpace. Hybrid Forman/Ollivier curvature with hardware-calibrated budget (Synthetic Transport Probe). Three-zone classifier v3 (κ_mean + Gini(PageRank) + η_F) stamps each graph GREEN/YELLOW/RED before pipeline starts. η_F = σ_F / √(2⟨k⟩) — dimensionless entropy index normalized against Poisson noise floor of Erdős-Rényi random graph with same mean degree. Threshold η=0.70 selected from clean gap [0.60, 0.76] in corpus: all YELLOW graphs (Grid, Ladder, Planar, Möbius) have η < 0.60, all RED graphs (ER, Bipartite) have η > 0.76. Validated at 97% accuracy on 35-graph corpus. Pre-runtime overhead: P50=56ms, MAX=125ms.

**Примечание Phase 3 (22.03.2026):** Четыре эксперимента Phase 3 завершены. Exp14 (anchors): grid divergence=0 (PASS), graph/tree >0.20 (FAIL) — local update вносит structural drift на нерегулярных пространствах. Exp15 (LCA-distance): Spearman r<0.3 во всех пространствах — дерево не семантично. Exp15b (bushes): silhouette>0.4 (кластеры есть), ARI≤0.210 (не стабильны). Exp16 (C-pre): gap>1.0, sil>0.3 все пространства — Track C UNFREEZE.

**Примечание Enox (21.03.2026):** Четыре observation-only инфраструктурных паттерна из Enox framework, реализованных под нужды проекта: (1) RegionURI — SHA256-адрес юнита, (2) DecisionJournal — append-only лог решений, (3) MultiStageDedup — 3-уровневая дедупликация (заготовка для multi-pass Phase 3, epsilon=0.0 → не срабатывает), (4) PostStepSweep — поиск идентичных sibling'ов в дереве. Все паттерны — чистая аннотация, не модифицируют state. Дефолты = False, zero overhead. Baseline fingerprint: 20 runs, DET-1 PASS. Интеграция завершена. NO REGRESSION (15/20 bitwise SAME).

**Примечание:** §A/B/C — секции плана валидации, написанного между Exp0.3 и Phase 1.
В §B «B1/B2» = probe-сцены. В P1 ниже «B1/B2/B3» = компрессия дерева. Контексты разные.

---

# Закрыто (не требует экспериментов)

| # | Вопрос | Статус | Источник |
|---|--------|--------|----------|
| 1 | Adaptive refinement работает? | **Да** | PoC, Exp0.1–0.2 |
| — | Halo обязателен? | **Да**, w ∈ [2,4], cosine feather | Exp0.2–0.3, Phase 1 |
| — | Probe обязателен? | **Да**, uncert, 5–10% бюджета | Exp0.8, Phase 2 |
| — | SeamScore как production-метрика | **Да**, dual check работает в 4 пространствах | Phase 2 |
| — | Governor (EMA) для бюджета | **Да**, StdCost −50%, penalty −85% | Exp0.8 |
| 2 | Комбинированная интересность нужна? | **Да, при деградации сигнала.** Двухстадийный гейт | Exp0.4–0.7b |
| 3 | Phase schedule по глубине? | **Нет** при текущих условиях | Exp0.8v5 |
| — | Halo cross-space applicability | **Правило выведено** (grid/graph: да, tree: нет). boundary parallelism >= 3 AND no context leakage | Phase 0 |
| — | Morton layout (element-level sort) | **Убит** (12-15× overhead, zero compute benefit). Но Morton как tile key encoding для lookup — жив, см. exp10e | 0.9a sandbox |
| — | Block-sparse layout (element-level B=8) | **Убит** (expansion ratio). Но paged sparse tiles — жив как кандидат в exp10e | 0.9a sandbox |
| — | Compact-with-global-reverse-map | **Убит** (VRAM +38.6%). Dense bookkeeping убивает sparse compute выигрыш | exp10 |
| — | P2a: ручные пороги гейта | **Ок**, ridge 100%, P2b не нужен | P2a sweep |
| — | DET-1: seed determinism | **PASS** (240/240 побитовое совпадение CPU+CUDA) | exp10d |
| — | Детерминизм non-overlapping writes | **Чисто** (bitwise match) | 0.9a sandbox |

---

# Иерархия экспериментов — детальное описание уровней (все до P3.5 CLOSED)

## Уровень 0: инфраструктурные предусловия

Без ответов на эти вопросы результаты всего выше — декоративные.

### P0. Layout на GPU

Зависимостей вверх нет — это фундамент.

```
P0. Layout на GPU
├── 0.9b0 (exp10): buffer-scaling probe — grid vs compact-with-reverse-map
│         РЕЗУЛЬТАТ: KILL compact (VRAM +38.6%). Grid — baseline.
│         Но: убита реализация, не принцип sparse. Compute O(k) быстрее O(M).
│
├── 0.9b1 (exp10e): tile-sparse layouts без global reverse_map
│         РЕЗУЛЬТАТ:
│         A (bitset): ALIVE — time -27..31%, VRAM +18%. Не sparse-memory, а execution layout.
│         B (packed Morton + binary search): KILLED по времени (+1700%). Storage idea жива.
│         C (paged): KILLED (+5000-30000%). Мёртв окончательно.
│
├── 0.9b2 (exp10f): packed tiles + alternative lookup
│         D passes Contour A, fails Contour B (peak VRAM from conv2d workspace).
│         E archived as contingency.
│
├── 0.9b3 (exp10g): dual-mode benchmark
│         РЕЗУЛЬТАТ: D_direct PASS both contours. -54% to -80% time, -36% to -86% peak VRAM.
│         Manual stencil (Contour A) + conv2d (Contour B) both pass.
│
├── 0.9b3 (exp10h): cross-space D_direct (vector_grid + tree_hierarchy)
│         РЕЗУЛЬТАТ: vector_grid 72/72 PASS both contours. tree 0/108 FAIL.
│         Tree failure: configs too small, per-level tile_map not amortized → exp10j.
│
├── 0.9b3 (exp10i): graph block-based addressing (3 partition strategies)
│         РЕЗУЛЬТАТ: Spatial graphs (random_geometric, grid_graph) conditionally viable
│         with spatial partition, cbr≤0.35. Scale-free (barabasi-albert) REJECTED, cbr=0.66.
│
├── 0.9b3 (exp10j): per-level tree break-even (158K trials)
│         РЕЗУЛЬТАТ: matmul op: D wins at p_l<0.375-0.40 for ALL level sizes.
│         stencil op: D saves memory but NEVER wins on time. Contour B: 45% PASS.
│         Policy: D_direct per-level only when operator is compute-heavy AND p_l < 0.40.
│
├── 0.9b:  end-to-end pipeline (после exp10g)
│         финалист vs grid, 4 пространства × 2 бюджета
│
└── 0.9h:  ⟶ ПОГЛОЩЁН DET-1 (✅ PASS)
```

**Принцип:** адресация на уровне абстракции операции. Refinement оперирует тайлами —
адресация должна быть tile-level, не element-level. Sparse снаружи, dense внутри тайла.

**Текущий статус P0: CLOSED.**

Final layout policy (all space types resolved):

| Space type | Layout | Status | Evidence |
|------------|--------|--------|----------|
| scalar_grid | D_direct (packed tiles + direct tile_map) | Production | exp10g: both contours PASS |
| vector_grid | D_direct (packed tiles + direct tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid: D_direct per-level where p_l<0.40 + matmul op; A_bitset elsewhere | Validated | exp10j: break-even found |
| irregular_graph / spatial | D_blocked (graph block addressing) conditional | Conditional | exp10i: spatial partition, cbr≤0.35 |
| irregular_graph / scale-free | A_bitset (dense grid + bitset mask) fallback | Fallback only | exp10i: blocks rejected, cbr=0.66 |

Layout naming glossary:
- **D_direct** = packed tiles + direct tile_map (O(1) lookup, no element-level reverse_map)
- **A_bitset** = dense grid + bitset mask (simple fallback)
- **D_blocked** = graph block addressing (block_map[block_id] -> slot, spatial graphs only)
- **E_hash** = hash table lookup (archived, dominated by D_direct)

Killed forever:
- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Binary search lookup on GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash as primary lookup (exp10f-E: dominated by D_direct)
- Fixed-size blocks for scale-free graphs (exp10i: cbr 0.64-0.99)

**Выход P0:** layout зафиксирован per space type. D_direct — production для grid.
Hybrid D_direct/A_bitset per-level — для tree. D_blocked conditional — для spatial graph.

### DET. Детерминизм и воспроизводимость (v1.8)

Кросс-инфраструктурное требование. Без DET-1 невозможен stability pass (Instrument Readiness Gate). Без DET-2 невозможен Track B.

```
DET. Детерминизм
├── DET-1: Seed determinism (Hard Constraint)
│         Два прогона, идентичные входы + seed → побитовое совпадение дерева.
│         CPU и GPU отдельно.
│         Компоненты: canonical traversal order (Z-order tie-break),
│                     deterministic probe (seed = f(coords, level, global_seed)),
│                     governor isolation (EMA update after full step).
│         Kill criterion: любое расхождение = fail.
│         Поглощает 0.9h (halo overlap determinism) как частный случай.
│
└── DET-2: Cross-seed stability (Soft Constraint)
          N=20 seeds × 4 пространства × 2 бюджета.
          Метрики: PSNR, cost, compliance, SeamScore.
          Kill criterion: CV > 0.10 для любой метрики = fail.
          (τ_cv=0.10 предварительный, уточняется по baseline.)
```

**Зависимости:** DET-1 зависит от P0 (layout определяет порядок обхода). DET-2 зависит от DET-1 (детерминизм — предусловие осмысленного измерения устойчивости).

**Выход DET:** подтверждение тестируемости (DET-1) и воспроизводимости (DET-2). Без этого Instrument Readiness Gate не проходится.

---

## Уровень 1: представление маршрутов

Зависит от P0 (layout определяет, как active_idx попадает в pipeline). Не зависит от "смысла" дерева — чистая инженерия хранения.

### P1. Компрессия и обслуживание структуры

```
P1. Компрессия дерева / маршрутов
├── B2: dirty-сигнатуры (12 бит: seam_risk + uncert + mass)
│       debounce (2 последовательных попадания)
│       сценарии: шум / структурное событие / drift
│       метрики: blast radius, latency-to-trigger, burstiness
│       ↑ это фундамент — без него B1 и B3 не знают когда запускаться
│
├── B1: segment compression (degree-2 + signature-stable + length cap)
│       зависит от B2 (критерий склейки = стабильность сигнатуры)
│       метрики: память vs node-per-node, стоимость локальных апдейтов
│
└── B3: anchors + periodic rebuild
        зависит от B1 + B2
        сценарий: частые локальные апдейты
        сравнение: (a) только локально (b) локально + periodic rebuild
        метрики: суммарная стоимость за N шагов, накопление "грязи"
```

**Текущий статус P1: CLOSED.** exp11 (dirty sig) PASS, exp13 (compression) PASS, exp14 (anchors) CONDITIONAL (grid PASS, graph/tree FAIL).

**Выход P1:** формат хранения дерева (flat nodes vs segments), механизм dirty detection, стратегия rebuild.

---

## Уровень 2: надёжность комбинированного сигнала

Зависит от P0 (pipeline работает), не зависит от P1 (компрессия ортогональна).

### P2. Автонастройка и устойчивость ρ

Двухстадийный гейт подтверждён (Exp0.7b), но пороги (instability, FSR) — ручные.

```
P2. Автонастройка ρ-гейта
├── P2a: sensitivity analysis — sweep порогов instability/FSR
│        на существующих сценах (clean/noise/blur/spatvar/jpeg)
│        вопрос: насколько плоский "хребет" оптимальности?
│        если широкий → ручные пороги ок, автонастройка не нужна
│        если узкий → нужен adaptive threshold
│
└── P2b: adaptive threshold (только если P2a показал узкий хребет)
         online estimation instability/FSR percentiles
         метрики: PSNR stability across scenes, overhead
```

**Текущий статус P2: CLOSED.** P2a PASS (ridge 100%), P2b not needed.

**Выход P2:** либо "ручные пороги ± 30% — без разницы" (и вопрос закрыт), либо конкретный механизм автонастройки.

---

## Уровень 3: семантика дерева

Зависит от P0 (layout) + P1 (формат хранения).

### P3. Даёт ли дерево смысловую метрику?

```
P3. Семантика дерева
├── P3a: LCA-расстояние как feature
│        на реальном дереве из pipeline: коррелирует ли LCA-distance
│        с "семантической близостью" (‖feature_i − feature_j‖)?
│        если нет → дерево = только журнал, не метрика
│
├── P3b: кусты (bushes) — кластеры путей
│        есть ли естественные кластеры среди leaf-путей?
│        метрики: silhouette, stability across runs
│
└── C-pre: проверка дискретности "профилей"
           trajectory features (EMA квантили, split signatures, stability)
           вопрос: есть ли кластерная структура?
           если да → C размораживается
           если нет → C мёртв
```

**Текущий статус P3: CLOSED.** P3a (exp15) FAIL, P3b (exp15b) FAIL, C-pre (exp16) PASS. Tree is a log, not a metric. Track C UNFROZEN.

**Выход P3:** либо "дерево — чисто журнал, смысловую метрику не даёт" (ок, не страшно), либо конкретный способ извлечения семантики.

---

## Уровень SC: Scale-Consistency Invariant (v1.7)

Частично формализует мета-вопрос v1.5 «как не сломать фичи». Зависит от P0 (pipeline), не зависит от P1/P2/P3. Может идти параллельно.

### SC-baseline. Верификация метрик D_parent / D_hf

```
SC-baseline. Scale-Consistency Verification
├── SC-0: фиксация пары (R, Up), проверка идемпотентности R              ✅ COMPLETE
├── SC-1: подготовка positive (strong + empirical) и negative baseline    ✅ COMPLETE
├── SC-2: вычисление D_parent, D_hf по всем случаям                      ✅ COMPLETE
├── SC-3: анализ separability (AUC, effect size, quantile separation)     ✅ COMPLETE
│         глобально + по уровням + по типам структуры
├── SC-4: kill criterion — PASSED с обновлённой формулой D_parent
│         (R=gauss σ=3.0, lf_frac normalization, AUC=0.853, d=1.491)     ✅ COMPLETE
└── SC-5: установка data-driven τ_parent[L] — нужна data-driven настройка порогов
```

**Kill criterion:** Global ROC-AUC >= 0.75, Depth-conditioned AUC >= 0.65, Effect size >= medium (d >= 0.5). Если не проходит — менять метрики, **не** подгонять пороги.

**SC-4 результат:** PASSED. Обновлённая формула D_parent: `||R(delta)|| / (||delta|| + epsilon)`, R=gauss sigma=3.0. AUC=0.853, d=1.491. Cross-space: T1=1.000, T2=1.000, T3=1.000, T4=0.824 (все >= 0.75).

**SC-5 статус:** tau_parent нуждается в data-driven threshold setting.

**Выход SC-baseline:** валидированные пороги tau_parent[L] или решение о пересмотре конструкции метрик.

Полный протокол: `docs/scale_consistency_verification_protocol_v1.0.md`.

### SC-σ sweep. Оптимизация параметра σ оператора R

```
SC-σ. Fine-grained sweep параметра σ
├── σ sweep: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
│       на каждом σ: полный SC-3 (AUC, Cohen's d, per-negative-type)
├── tile_size sweep: σ × tile_size ∈ {8, 16, 32, 64}
│       вопрос: σ_opt зависит от tile_size? Есть ли σ/tile_size ≈ const?
├── cross-space: σ sweep × 4 типа пространств
│       вопрос: σ_opt одинаков для всех пространств или space-dependent?
└── вывод: формула/правило для выбора σ, или фиксированный σ_opt
```

**Known limitation текущего σ=3.0:** выбрано как наименьшее целое в грубом sweep [0.5, 1.0, 2.0, 3.0]. Fine-grained поиск не проводился. Возможно σ=2.5 достаточно, или σ=4.0 лучше. Оптимум может зависеть от tile_size и типа пространства.

**Зависимости:** нет (можно запускать параллельно с чем угодно, переиспользует код sc_baseline).
**Приоритет:** низкий (σ=3.0 проходит kill criteria; оптимизация — не блокер).

---

### SC-enforce. Enforcement (после SC-baseline) — ✅ CLOSED (exp14a)

```
SC-enforce. Scale-Consistency Enforcement
├── damp delta / reject split / increase local strictness при D_parent > τ_parent
└── D_parent как контекстный сигнал в ρ (не самодостаточный)
```

**Результат (exp14a_sc_enforce):** Three-tier enforcement (pass/damp/reject) + strictness-weighted waste budget + adaptive τ T4(N) = τ_base * (1 + β/√N). Adaptive tau resolves high reject rate (~50%) on tree_hierarchy with tight T4 thresholds.

---

## Уровень 4: глобальная согласованность ("не сломать фичи")

Зависит от **всего выше** + SC-baseline. Мета-вопрос из Concept v1.5, частично формализован через Scale-Consistency Invariant (Concept v1.7, раздел 8).

### P4. Согласованность представления при неоднородной глубине

```
P4. "Не сломать фичи"
├── P4a: downstream consumer test
│        подать adaptive-refined представление в простой downstream
│        (классификатор / автоэнкодер)
│        сравнить с dense-refined и coarse-only
│        вопрос: ломается ли downstream при неоднородной глубине?
│        (с enforcement scale-consistency vs. без)
│
├── P4b: matryoshka invariant
│        проверить что представление на любом уровне "матрёшки"
│        валидно как вход для потребителя
│        (не только визуально гладко, но функционально корректно)
│
└── P4c: механизм гарантии (если P4a/b показали проблему)
         варианты: padding/projection слой, consistency loss,
         depth-aware normalization, усиление τ_parent
```

**Выход P4:** либо "неоднородная глубина не ломает downstream" (и вопрос закрыт), либо конкретный механизм защиты.

### P5-noise. Robustness к шумным данным

**Обнаружено:** viz testbed (24 марта 2026). Система оптимизирует к наблюдаемым данным, не к истинному сигналу. Все эксперименты P0-P4 на чистой синтетике.

```
P5-noise. Noise robustness
├── exp20a: sweep подходов к noise-aware refinement
│        Кандидаты (все на T1 scalar grid, 3 уровня σ × 10 seeds):
│        (A) Smoothed ρ — ρ от сглаженных данных, refinement = raw
│        (B) Noise floor (Morozov) — skip tile if residual < σ√N
│        (C) BayesShrink — analytical soft threshold в wavelet domain
│        (D) SureShrink — SURE-optimized threshold, model-free
│        (E) Coarse-as-prior — refined = α×observed + (1-α)×coarse, α = σ²_s/(σ²_s+σ²_n)
│        (F) SURE-Bayes blend — (E) + SURE для α-selection
│        Метрики: MSE-to-clean (oracle), PSNR, SURE risk, compute cost
│        Kill: oracle MSE > coarse MSE (refinement ухудшает quality)
│        σ² estimation: MAD на Лапласиане (grids), TBD для graph/tree
│
├── exp20b: composite из лучших
│        Собрать лучшие A-F по Pareto (quality × cost)
│        Возможные композиты:
│        - Smoothed ρ (selection) + BayesShrink (refinement)
│        - Noise floor (gate) + Coarse-as-prior (refinement)
│        - Any winner + SURE spot-check (5-10% тайлов как audit)
│        Валидация: T1 + T2 + T3(spatial) + T4
│        σ² estimation для T3/T4 — research question
│
└── exp20c: интеграция в pipeline
         Финальный композит → harness, governor-aware
         DET-1/DET-2 re-check с noise
```

**Выход P5-noise:** noise-aware refinement operator + σ² estimator per space type + updated gate thresholds.

**Зависимость:** после P4 (нужен стабильный pipeline). Но Phase 4 эксперименты на шумных данных запрещены без noise-awareness.

---

# Заморожено

## C. DAG + профили — РАЗМОРОЖЕН (22.03.2026)

**Входной контракт (все три одновременно):**

1. Минимум две несводимые цели (не сводятся в скаляр без потери семантики)
2. Конкретный downstream consumer, который на этих целях реально стоит
3. Наблюдаемый конфликт: разные оптимальные решения при разных целях на одних данных

**C-pre результат (exp16):** PASS. Gap > 1.0 и Silhouette > 0.3 во всех 4 пространствах. Дискретные профили существуют. Track C РАЗМОРОЖЕН. Однако входной контракт (пункты 1-3) всё ещё требует конкретизации для следующих шагов.

---

# Граф зависимостей

```
P0 (layout GPU)
 ├──→ DET-1 (seed determinism) ──→ DET-2 (cross-seed stability)
 │         │
 ├──→ P1 (компрессия дерева)  ──→ P3 (семантика дерева)
 │    └── exp11 (dirty sig) ──→ exp13 (segment compression)
 │                                       │
 ├──→ P2 (автонастройка ρ)               ├──→ C-pre
 │                                        │
 └──→ SC-baseline (✅ SC-0..SC-4) ──→ SC-5 (exp12a) ──→ SC-enforce (exp14a) ──→ P4
                                                                │
                                                                ▼
 exp07 + exp10d + exp12a + exp14a ──→ exp_phase2_pipeline ──→ exp_phase2_e2e (✅ 240 configs, DET-1)
                                                                │
                                                                ▼
 exp_phase2_e2e + exp14 (FAIL → rho decomposition) ──→ exp17 (three-layer rho, Phase 3.5)
```

**Критический путь:** P0 → DET-1 → P1 → P3 → P4.

**Phase 2 путь (✅ CLOSED):** exp07 + exp10d + exp12a + exp14a → exp_phase2_pipeline → exp_phase2_e2e.

**Phase 3.5 путь:** Phase 2 pipeline + Phase 3 results (exp14 FAIL motivated rho decomposition) → exp17 (three-layer rho).

**Параллельные ветки:** P2, SC-baseline — обе идут параллельно P1, все нужны до P4. DET-2 параллельно P1 (после DET-1).

**Gate-блокеры:** DET-1 блокирует stability pass. DET-2 блокирует Track B.

---

# Рабочий порядок

Все пункты до Phase 4 включительно завершены (exp01--exp19).

1. ~~**P0: 0.9b0** — buffer-scaling probe, kill/go для compact~~ ✅
2. ~~**P0: 0.9b/0.9c** — если compact жив; иначе фиксируем grid~~ ✅
3. ~~**DET-1** — seed determinism~~ ✅
4. ~~**P1-B2** — dirty-сигнатуры~~ ✅
5. ~~**DET-2** — cross-seed stability~~ ✅
6. ~~**P2a** — sensitivity sweep порогов гейта~~ ✅
7. ~~**SC-5** — установка data-driven τ_parent[L]~~ ✅
8. ~~**P1-B1** — segment compression~~ ✅
9. ~~**P1-B3** — anchors + rebuild~~ ✅ (CONDITIONAL: grid PASS, graph/tree FAIL)
10. ~~**SC-enforce** — enforcement scale-consistency~~ ✅
11. ~~**P3a/P3b** — семантика дерева~~ ✅ (FAIL — дерево не семантично)
12. ~~**C-pre** — кластерность профилей~~ ✅ (PASS — Track C UNFREEZE)
13. ~~**Phase 3.5 (exp17)** — трёхслойная декомпозиция ρ~~ ✅ (reusability 12/12 PASS)
14. ~~**P4 (exp19)** — multi-tick sweep (2050 конфигов, 5 sub-экспериментов)~~ ✅ (multi-tick +2-7% under noise, ROI fix)
15. **P5-noise** — denoising refinement (exp20) — NEXT

---

# Конвенция нумерации (v3+)

Исторически нумерация росла стихийно: хронологические номера (0.1–0.9a),
буквенные секции плана валидации (§A/B/C), уровни roadmap (P0–P4),
суб-эксперименты внутри уровней (B1–B3 в P1). Результат — путаница.

**Правила для новых экспериментов:**

1. **Единая сквозная нумерация.** Следующий эксперимент = `exp10`.
   Нумерация целочисленная, без точек (точка путалась с подверсиями).
   Номер = порядок создания. Никогда не переиспользуется.

2. **Суб-эксперименты — строчная буква.** `exp10a`, `exp10b`, `exp10c`.
   Одна серия = один числовой корень.

3. **Папка = `exp{N}{суффикс}_{краткое_имя}/`.** Например:
   `exp10_buffer_scaling/`, `exp10a_synthetic_kernel/`, `exp11_dirty_signatures/`.

4. **Маппинг на roadmap — только в этом документе**, не в именах папок.
   Папка не содержит «P0» или «B2» в названии.

5. **Каждая папка содержит README.md** (короткий, 5–15 строк):
   - Вопрос/гипотеза (одно предложение)
   - Kill criteria
   - Ссылка на уровень roadmap (P0/P1/P2/...)
   - Статус (open / closed / killed)

6. **Старые имена не переименовываются.** `phase1_halo/`, `phase2_probe_seam/`,
   `exp09a_layout_sandbox/` — историческое наследие, связь с новой нумерацией
   зафиксирована в таблице маппинга выше.

**Маппинг рабочего порядка → номера экспериментов:**

| Шаг | Roadmap | Описание | Будущий exp# |
|-----|---------|----------|-------------|
| 1 | P0 | buffer-scaling probe (kill/go compact) | exp10 |
| 2 | P0 | end-to-end pipeline grid vs compact | exp10a/b/c |
| 3 | DET-1 | seed determinism (canonical order, det. probe, governor isolation). Поглощает 0.9h | exp10d |
| 4 | P1-B2 | dirty-сигнатуры | exp11 |
| 5 | DET-2 | cross-seed stability (20 seeds × 4 пространства × 2 бюджета) | exp11a |
| 6 | P2a | sensitivity sweep порогов гейта (5 сцен × 4 пространства) | exp12 |
| 7 | SC-5 | установка data-driven τ_parent[L] (SC-0..SC-4 ✅) | exp12a |
| 8 | P1-B1 | segment compression (thermodynamic guards, N_critical=12) | exp13 ✅ |
| 9 | P1-B3 | anchors + rebuild (grid PASS, graph/tree FAIL — structural drift) | exp14 ✅ CONDITIONAL |
| 10 | SC-enforce | enforcement: three-tier + waste budget + adaptive τ | exp14a ✅ |
| — | SC-σ | fine-grained σ sweep × tile_size × 4 пространства (низкий приоритет) | exp14b | (deferred)
| 10½ | Phase 2 | full pipeline assembly (gate+governor+SC-enforce+probe+traversal) | exp_phase2_pipeline ✅ |
| 10¾ | Phase 2 | end-to-end validation (4 spaces, 240 configs, DET-1 verified) | exp_phase2_e2e ✅ |
| 10⅞ | Topo | topological pre-runtime profiling: hybrid Forman/Ollivier curvature + three-zone classifier v3 (κ+Gini+η_F). 35-graph corpus, 97% accuracy. η_F=0.70 from gap [0.60, 0.76] | exp_phase2_pipeline (topo_features.py) ✅ |
| 11 | P3a | LCA-distance vs feature similarity (Spearman r<0.3 all spaces) | exp15 ❌ FAIL |
| 11b | P3b | bush clustering (sil>0.4 but ARI≤0.210 — unstable) | exp15b ❌ FAIL |
| 12 | C-pre | trajectory profile clustering (gap>1.0, sil>0.3 — Track C UNFREEZE) | exp16 ✅ PASS |
| 13 | Three-layer rho | Decompose monolithic rho into L0 (topology) → L1 (presence) → L2 (query). Cascade quotas, streaming, industry benchmarks. | exp17 ✅ PASS |
| 14 | RG-flow | Basin membership vs feature similarity (r=0.019, FAIL — deferred to post-multi-pass) | exp18 ❌ FAIL (deferred) |
| 15 | P4 | Multi-tick sweep: scaling, gate stress, param sweep, real data, noisy/hetero (2050 configs) | exp19 ✅ DONE |
| 16 | P5-noise | Noise robustness: sweep подходов к denoising при refinement, затем композит из лучших | exp20 (exp20a sweep, exp20b composite) |

Все пункты до exp19 включительно завершены (exp01--exp19). exp18 deferred to post-Phase 4.
exp20 (noise robustness) — Phase 5, но awareness и baseline нужны уже в Phase 4.

Номера предварительные. Если между шагами возникнет незапланированный
эксперимент — он получает следующий свободный номер.

---

# Instrument Readiness Gate

Все эксперименты P0–P4 + SC + DET относятся к **Track A** (построение инструмента). Переход к **Track B** (исследование структуры дерева) — только после прохождения Instrument Readiness Gate:

1. **Invariant pass** — все обязательные инварианты выполняются (включая DET-1: seed determinism)
2. **Overhead profile** — overhead не съедает выигрыш
3. **Stability pass** — DET-1 (побитовое совпадение при фиксированном seed) + DET-2 (CV метрик < τ_cv по seeds)
4. **One validated benchmark** — adaptive > random > coarse с подтверждёнными числами
5. **Attribution diagnostics** — вклад каждого модуля измерен (ablation)

Подробно: `docs/target_problem_definition_v1.1.md`.

После успешной Track B открывается **Track C** (обобщение на нон-пространственные домены: графы, латентные пространства, активации). Долгосрочная амбиция, не текущая цель.

---

# Принципы

* Сначала судья-цифры, потом амбиции.
* Ни один "следующий этап" не фиксируется заранее.
* Kill criteria — двусторонние (speed + memory).
* Forensic-grade протокол: controls, Holm-Bonferroni, cost-fair comparisons.

---

# Experiment Hierarchy (English Summary)

This document tracks experiment status, dependencies, and execution order.

All experiments through exp19 (Phase 4) are complete.

## Completed Levels

| Level | Topic | Status | Key Result |
|-------|-------|--------|------------|
| P0 | GPU Layout | CLOSED | D_direct for grid/vector, hybrid for tree, D_blocked conditional for spatial graph |
| DET-1 | Seed Determinism | PASS | 240/240 bitwise match CPU+CUDA |
| DET-2 | Cross-seed Stability | PASS | 8/8 per-regime CV thresholds |
| P1 | Tree Compression | CLOSED | Dirty signatures (exp11), segment compression (exp13), anchors (exp14: grid PASS, graph/tree FAIL) |
| P2 | Rho-gate Tuning | CLOSED | Ridge width 100%, manual thresholds OK |
| SC | Scale-Consistency | CLOSED | D_parent AUC=0.853, per-space tau, three-tier enforcement |
| P3 | Tree Semantics | CLOSED | LCA-distance FAIL, bushes FAIL, C-pre PASS (Track C UNFREEZE) |
| Phase 2 | Pipeline Assembly | CLOSED | 240 configs E2E, topo profiling 97% accuracy |
| Phase 3.5 | Three-Layer Rho | CLOSED | exp17: 1080 configs, reusability 12/12 PASS (min 0.838), cascade quotas, streaming |
| Phase 4 | Multi-tick Pipeline | CLOSED | exp19: 2050 configs, multi-tick +2-7% under noise, ROI fix, scaling law validated |

## Open

| Level | Topic | Status |
|-------|-------|--------|
| P4a | Downstream consumer test ("don't break features") | NEXT |
| P5-noise | Noise robustness: sweep denoising approaches (exp20a), composite (exp20b), integration (exp20c) | After P4. Discovered via viz testbed 24 Mar 2026 |
| Track C | DAG + profiles (UNFROZEN by exp16) | Entry contract needs concretization |
| SC-sigma | Fine-grained sigma sweep | Low priority (sigma=3.0 passes kill criteria) |

## Dependency Graph (simplified)

```
P0 -> DET-1 -> DET-2
P0 -> P1 (exp11, exp13, exp14) -> P3 (exp15, exp15b, exp16)
P0 -> SC-baseline -> SC-5 (exp12a) -> SC-enforce (exp14a) -> P4
Phase 2 pipeline -> Phase 2 E2E -> exp17 (Phase 3.5)
All above -> P4 (exp19 DONE, 2050 configs)
P4 -> P5-noise (exp20a sweep, exp20b composite, exp20c integration) (NEXT)
P4 -> Track C (after entry contract concretization)
```

## Naming Convention

Sequential integer numbering: exp10, exp11, ..., exp19. Sub-experiments use lowercase suffix: exp10a, exp10b, exp19a-e. Folder format: `exp{N}{suffix}_{short_name}/`.
