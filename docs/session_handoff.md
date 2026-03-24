# Session Handoff — Curiosity (Phase 4 next)

Документ для новой сессии AI-оркестратора. Содержит полный контекст для немедленного продолжения работы.

---

## English Summary

**Current status:** Phases 0–3.5 DONE. Next up: Phase 4.

- **Phase 0** (18 Mar 2026): Environment setup, halo cross-space validation, SC-baseline.
- **Phase 1** (20 Mar 2026): P0 Layout closed (D_direct for grids, hybrid for trees, D_blocked for spatial graphs, A_bitset fallback). DET-1 and DET-2 PASS.
- **Phase 2** (20 Mar 2026): End-to-end pipeline assembled and validated. SC-enforce integrated. Topo profiling integrated.
- **Enox infrastructure** (21 Mar 2026): Four observation-only patterns (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep).
- **Phase 3** (22 Mar 2026): Exp14 anchors (grid PASS, graph/tree FAIL). Exp15 LCA-distance FAIL. Exp15b bushes FAIL. Exp16 C-pre PASS, Track C UNFREEZE.
- **Phase 3.5** (23 Mar 2026): Three-layer rho decomposition (L0 topology, L1 presence with cascade quotas, L2 query). Streaming pipeline. 1080 configs, reusability 12/12 PASS. Industry benchmarks.
- **Exp18** (23 Mar 2026): Basin membership FAIL (r=0.019). RG-flow basins don't form in single-pass. Deferred to post-multi-pass.
- **Budget control architecture:** Three orthogonal mechanisms: (1) L1 cascade quotas (structural), (2) Governor EMA (hardware-adaptive, batch/reuse only — NOT streaming), (3) WasteBudget + StrictnessTracker (self-tightening noose, safety). Streaming uses Institutional Inequality Formula: W = N×(1-ECR)^γ + forward carry. γ sweep planned: {1.0..4.0}.
- **Runtime visualization** (24 Mar 2026): Interactive testbed `viz/index.html` — real multi-tick T1 runtime with per-tile strictness, two-stage gate with pilot-calibrated thresholds, EMA weight adaptation, noise injection. Exposed **9 architectural issues** recorded in `docs/workplan.md`: (1) missing convergence detector, (2) gate health thresholds not data-driven, (3) FSR inflated by refined tiles, (4) gate oscillation without hysteresis, (5) EMA weights uncalibrated on first ticks, (6) gain/cost incommensurability, (7) probe/reject overlap, (8) no post-refinement quality feedback, (9) **noise-fitting: system optimizes to noisy observations, not true signal** — fundamental blind spot, all experiments ran on clean synthetic data so never caught. Phase 5 (robustness) but awareness needed in Phase 4.
- **Next: Phase 4** — P4a downstream consumer test, P4b matryoshka, MultiStageDedup test, Governor EMA restoration + sweep (3 modes × 3 hw × 6 γ × 4 spaces × 20 seeds = 4320 configs). **Must address 8 runtime issues (1-8) from viz testbed before sweep. Issue 9 (noise-fitting) is Phase 5 but Phase 4 experiments must not use noisy data without noise-awareness.** Bushes revisit after Track C. RG-flow after multi-pass.

**Working PC:** PC 2 (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Working directory: `R:\Projects\Curiosity`.

---

## Где мы

Проект Curiosity.

- Фаза 0 **завершена** (18 марта 2026).
- Фаза 1 **завершена** (20 марта 2026). Все потоки — PASS. P0 Layout **ЗАКРЫТ**. DET-1 **PASS**. DET-2 **PASS**.
- Фаза 2 **завершена** (20 марта 2026). Pipeline assembled, SC-enforce integrated, E2E validated.
- **Enox-инфраструктура** — ✅ DONE (21 марта 2026). Четыре observation-only паттерна (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep) — чистая аннотация, не меняют pipeline state.
- Фаза 3 **завершена** (22 марта 2026). Exp14 anchors: grid PASS, graph/tree FAIL. Exp15 LCA-distance: FAIL. Exp15b bushes: FAIL. Exp16 C-pre: PASS → **Track C UNFREEZE**.
- Фаза 3.5 **завершена** (23 марта 2026). Exp17 three-layer rho: архитектурная декомпозиция ρ на L0 (topology) → L1 (presence, cascade quotas) → L2 (query). Streaming pipeline. 1080 конфигов, reusability 12/12 PASS. Industry benchmarks (kdtree, quadtree, wavelets, leiden).
- Exp18 **завершён** (23 марта 2026). Basin membership FAIL (r=0.019). RG-flow бассейны не формируются в single-pass. Deferred после multi-pass.
- **Бюджетный контроль:** три ортогональных механизма: (1) L1 cascade quotas (структурный), (2) Governor EMA (hardware-adaptive, batch/reuse only — НЕ streaming), (3) WasteBudget + StrictnessTracker (удавка, safety). Streaming: формула институционального неравенства W = N×(1-ECR)^γ + forward carry. γ sweep запланирован: {1.0..4.0}.
- **Визуализация рантайма** (24 марта 2026): интерактивный стенд `viz/index.html` — реальный multi-tick T1 runtime с per-tile strictness, двухступенчатым gate (pilot calibration + EMA-веса), инъекцией шума. Выявлено **9 архитектурных проблем**, записаны в `docs/workplan.md`: (1) нет convergence detector, (2) пороги gate не data-driven, (3) FSR раздувается от refined tiles, (4) осцилляция gate без hysteresis, (5) EMA-веса не откалиброваны на первых тиках, (6) несоизмеримость gain/cost, (7) probe пересекается с rejected, (8) нет обратной связи от качества refinement, (9) **noise-fitting: система оптимизирует к зашумлённым наблюдениям, а не к сигналу** — фундаментальный blind spot, все эксперименты были на чистой синтетике. Phase 5 (robustness), но Phase 4 должна быть noise-aware.
- **Следующий шаг — Фаза 4:** P4a downstream consumer test, P4b matryoshka, MultiStageDedup test, Governor EMA restoration + sweep (3 режима × 3 hw × 6 γ × 4 spaces × 20 seeds = 4320 конфигов). **Перед sweep обязательно закрыть 8 runtime issues (1-8) из viz стенда. Issue 9 (noise-fitting) — Phase 5, но Phase 4 эксперименты не запускать на шумных данных без noise-awareness.** Bushes revisit после Track C. RG-flow после multi-pass.

Рабочий ПК: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Рабочая директория: `R:\Projects\Curiosity`.

## Что читать (в этом порядке)

| # | Файл | Зачем |
|---|------|-------|
| 1 | `docs/session_handoff.md` | **Этот файл** — текущий статус, что делать дальше |
| 2 | `docs/concept_v2.0.md` | Каноническая концепция (актуальная) |
| 3 | `docs/teamplan.md` | План с отметками Фаза 0-3.5, описание Фаз 4+ |
| 4 | `docs/experiment_hierarchy.md` | Граф зависимостей, приоритеты, нумерация exp10+ |
| 5 | `docs/workplan.md` | Модули A-H, roadmap C-оптимизации |
| 6 | `docs/glossary.md` | Все термины проекта (обновлён 23.03.2026) |
| 7 | `docs/environment_2.md` | Как активировать .venv-gpu на PC 2 (CUDA) |

## Фаза 1 — Финальные результаты (20 марта 2026)

### Основные потоки

| Stream | Результат | Статус |
|--------|-----------|--------|
| S1 exp10 | KILL compact-with-reverse-map (VRAM +38.6%). Grid — baseline. | PASS |
| S1b exp10d | DET-1 PASS (240/240 побитовое совпадение CPU+CUDA) | PASS |
| S2 exp11 | PASS | PASS |
| S3 P2a | PASS — ridge 100%. Ручные пороги ok. P2b не нужен. | PASS |
| S4 exp12a | PASS | PASS |
| S5 deferred | Research note done. | PASS |
| DET-2 | PASS (cross-seed stability) | PASS |

**Gate Phase 1 -> Phase 2: PASSED.** Все потоки PASS. P0 Layout закрыт. DET-1 и DET-2 пройдены.

---

## P0 LAYOUT — ЗАКРЫТ (полная серия exp10, 19 марта 2026)

### Словарь layout-ов

- **D_direct** ("packed tiles + прямой tile_map") — активные тайлы в компактном массиве, tile_map[tile_id] -> slot для O(1) lookup. Без element-level reverse_map. Победитель для сеток.
- **A_bitset** ("плотная сетка + bitset маска") — полноразмерный тензор данных + битовая маска активации. Простой fallback.
- **D_blocked** ("блочная адресация для графов") — узлы графа разбиты на фиксированные блоки, block_map[block_id] -> slot. Работает только для пространственных графов.
- **E_hash** ("hash table lookup") — архивный fallback, доминируется D_direct на текущем масштабе. Триггеры воскрешения задокументированы.

### Финальная layout policy

| Тип пространства | Layout | Статус | Доказательство |
|-----------------|--------|--------|---------------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production | exp10g: оба контура PASS |
| vector_grid | D_direct (packed tiles + tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Гибрид: D_direct per-level где occupancy < 40% + тяжёлый compute; A_bitset остальное | Validated | exp10j: break-even найден |
| irregular_graph / spatial | D_blocked (блочная адресация) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (плотная сетка + bitset маска) fallback | Только fallback | exp10i: блоки отклонены, cbr=0.66 |

### Убито навсегда

- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Бинарный поиск на GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash как основной lookup (exp10f-E: доминируется D_direct)
- Фиксированные блоки для scale-free графов (exp10i: cbr 0.64-0.99)

---

## Что делать — Фаза 2

### Цель

End-to-end pipeline validation. Собрать весь pipeline (layout + halo + gate + governor + probe + SeamScore) и прогнать на реальных задачах.

### Задачи

1. **Интеграция pipeline** — собрать все валидированные компоненты в единый runtime:
   - Layout (D_direct / A_bitset / D_blocked / гибрид) по типу пространства
   - Halo (cosine feathering, overlap >= 3)
   - Двухстадийный гейт (residual-first + utility-weighted fallback)
   - Budget governor (EMA-контроллер strictness)
   - Probe (5-10% бюджета на exploration)
   - SeamScore (Jumpout / (Jumpin + eps))

2. **End-to-end валидация** — прогнать собранный pipeline на реальных задачах, все 4 типа пространств

3. **SC-enforce** — интеграция enforcement (damp delta / reject split при D_parent > tau_parent)

### Критический путь

```
Phase 2 → Instrument Readiness Gate → Track A
```

---

## Открытые вопросы для Фазы 2

1. **Budget Governor гистерезис для irregular high budget** — Track B. Как настроить StrictnessTracker + WasteBudget для пространств с нерегулярным бюджетным профилем. (Примечание: GovernorIsolation из exp10d — это EMA-телеметрия для DET-1, не бюджетный контроллер.)

2. **Graph-native sparse для scale-free** — Track C. CSR/COO вместо A_bitset fallback для scale-free графов (barabasi-albert и подобных).

3. **C(I,M,p) surface с правильными метриками** — Track C. Построение поверхности curiosity = f(information, mass, position) с корректными метриками для каждого типа пространства.

---

## Фаза 2 — Финальные результаты (20 марта 2026)

### Потоки

| Stream | Описание | Статус |
|--------|----------|--------|
| A (Pipeline Assembly) | CuriosityPipeline: gate + governor + SC-enforce + probe + traversal | ✅ DONE |
| B (SC-Enforce) | Three-tier pass/damp/reject + strictness-weighted waste budget + adaptive τ T4(N) = τ_base*(1+β/√N) | ✅ DONE |
| C (Segment Compression) | Thermodynamic guards (N_critical=12, bombardment guard) — eliminate overhead on small trees | ✅ DONE |
| D (E2E Validation) | 240 configs, 4 space types, DET-1 + DET-2 recheck with topo | ✅ DONE |

### Ключевые находки

- **tree_hierarchy high reject rate (~50%)** из-за тесных T4 thresholds — resolved with adaptive τ T4(N) = τ_base * (1 + β/√N)
- **Graph clustering** upgraded from k-means to Leiden (community detection), validated on 10 pathological topologies: Swiss Roll, Barbell, Hub-Spoke, Ring of Cliques, Bipartite, Erdos-Renyi, Grid, Planar Delaunay, Mobius strip
- **E2E irregular_graph перезапущен** (21.03.2026) с topo profiling active: zone distribution GREEN 75%/RED 25%, PSNR −0.20 dB vs pre-topo (ожидаемо)

### E2E результаты (с topo profiling, 21.03.2026)

| Space | PSNR gain median | Reject max | Wall max |
|-------|-----------------|------------|----------|
| scalar_grid | +7.34 dB | 0% | 23ms |
| vector_grid | +1.46 dB | 0% | 33ms |
| irregular_graph | +3.54 dB | 0% | 245ms |
| tree_hierarchy | +1.48 dB | 0% | 4ms |

### DET Rechecks (с topo profiling, 21.03.2026)

| Тест | Scope | Результат |
|------|-------|-----------|
| DET-1 (bitwise determinism) | 4 spaces × 10 seeds = 40 | ✅ 40/40 PASS |
| DET-2 (cross-seed stability) | 4 spaces × 20 seeds × 2 budgets = 160 | ✅ 8/8 cells PASS |

DET-2 kill metrics (n_refined, compliance): CV≈0. psnr_gain CV=0.09–0.37 — информационная, зависит от seed GT (не kill-criteria).

**Gate Phase 2 -> Phase 3: PASSED.** All streams DONE. Pipeline assembled and validated end-to-end. Topo profiling integrated and DET-verified.

### Enox-инфраструктура (21 марта 2026) — ✅ DONE

**Источник:** Enox open-source фреймворк. Взяты 4 паттерна (идеи, не код), реализованы под наши нужды.

**Принцип:** все паттерны — чистое наблюдение/аннотация. Никогда не модифицируют pipeline state. Все дефолты = False (backward compatible). DET-1 должен по-прежнему проходить.

**4 паттерна:**

| # | Паттерн | Назначение | Статус сейчас |
|---|---------|-----------|---------------|
| 1 | RegionURI | SHA256-адрес каждого юнита (parent_id\|op_type\|child_idx → 16 hex) | Готов |
| 2 | DecisionJournal | Append-only лог решений гейта/enforce с метриками | Готов |
| 3 | MultiStageDedup | 3-уровневая дедупликация (exact hash / metric / policy). epsilon=0.0 → не срабатывает в single-pass | Готов (заготовка для Phase 3) |
| 4 | PostStepSweep | Поиск идентичных sibling'ов в tree_hierarchy (merge candidates) | Готов |

**Config knobs (все default=False):**
`enox_journal_enabled`, `enox_dedup_enabled`, `enox_dedup_epsilon` (0.0), `enox_sweep_enabled`, `enox_sweep_threshold` (0.05), `enox_include_uri_map`

**Baseline fingerprint:** 20 runs (4 spaces × 5 seeds), budget=0.30. PSNR median +2.32 dB, DET-1 PASS, wall time median 11.9ms.

**Ключевые файлы:**
- `exp_phase2_pipeline/enox_infra.py` — реализация 4 паттернов
- `exp_phase2_pipeline/enox_comparison.py` — before/after тестовый фреймворк
- `exp_phase2_pipeline/config.py` — 6 новых ручек
- `exp_phase2_pipeline/pipeline.py` — интеграция хуков (DONE)

**Результат:** ✅ DONE
- Pipeline.py: все хуки интегрированы (journal после каждого решения, URI update, PostStepSweep, PipelineResult)
- Smoke test: PASS (4 spaces)
- Enox enabled test: journal/dedup/sweep/uri_map — все работают
- Comparison: NO REGRESSION (15/20 bitwise SAME, 5 DIFF = topo calibration)
- DET-1: PASS

### Топологический профайлинг графовых пространств (21 марта 2026)

**Проблема:** pipeline не видел топологию пространства до начала работы. ρ-функция распределяла бюджет вслепую — без учёта мостов, хабов, структурного хаоса.

**Решение:** многоступенчатый профайлинг при инициализации IrregularGraphSpace, ДО первого pipeline tick:

1. **Гибридная кривизна** — Forman-Ricci O(1) для ВСЕХ рёбер + Ollivier-Ricci (точный EMD через linprog HiGHS) для top-N аномальных рёбер. N = floor(topo_budget_ms / t_ollivier_ms), бюджет ≤ 50ms.
2. **Аппаратная калибровка** — Synthetic Transport Probe: κ_max = W_test · ∛(τ_edge / t_test). Одноразовый замер при старте сессии (52ms).
3. **Трёхзонный классификатор (v3)** — присваивает графу клеймо GREEN/YELLOW/RED:
   - **GREEN** (κ_mean > 0): плотные клики, ECR < 5%. Карт-бланш бюджету, ослабление τ_eff.
   - **YELLOW** (κ < 0, Gini < 0.12, η_F ≤ 0.70): однородные решётки, ECR 10-25%. Стандартные лимиты.
   - **RED** (κ < 0, Gini ≥ 0.12 ИЛИ η_F > 0.70): структурный хаос, ECR > 30%. Максимальное затягивание τ, блокировка глубокого расщепления.
4. **η_F = σ_F / √(2⟨k⟩)** — безразмерный индекс топологической энтропии. σ_F = дисперсия Forman-кривизны, √(2⟨k⟩) = шумовой предел дисперсии случайного графа Эрдёша-Реньи с той же средней степенью. Граф, у которого структурная дисперсия ниже пуассоновского пола — регулярный. Выше — фонит хаосом.

**Обоснование порога η_F = 0.70:**
- Threshold sweep по κ<0 подмножеству корпуса (22 графа):
  - YELLOW графы (Grid, Ladder, Planar, Möbius): η_F < 0.60 — чётко ниже порога
  - RED графы (ER, Bipartite): η_F > 0.76 — чётко выше
  - Gap [0.60, 0.76] — мёртвая зона, ни один граф корпуса туда не попадает
  - Порог 0.70 — середина gap, максимальный margin с обеих сторон
- Единственный пограничный случай: Watts-Strogatz (η_F = 1.02) → RED. Классифицирован корректно: WS с p=0.1 — кольцо с впрыснутым пуассоновским хаосом, η > 1.0 = структурный шум объективно выше случайного фона. То, что Leiden вырезает куски с ECR=15.7% — алгоритмическое везение, не свойство топологии.
- Стабильность: диапазон η_thresh ∈ [0.60, 0.75] даёт одинаковые 86% на κ<0 подмножестве. Порог не на лезвии.

**Валидация:** 35-graph corpus (9 базовых + 26 вариаций масштаба/параметров). v3 accuracy = 97% (34/35, единственный документированный граничный случай — Karate Club, 3 п.п. от порога).

**Производительность (pre-runtime, одноразовый setup):**
- P50 = 56ms, P90 = 85ms, MAX = 125ms (Swiss Roll 1000 узлов)
- Leiden: 1.2ms средняя (пренебрежимо)
- Topo features + classifier: 57.6ms средняя
- Overhead vs pipeline tick (500ms): 11.8% mean, 25% worst case

**Ключевые файлы:**
- `exp_phase2_pipeline/topo_features.py` — ядро: CalibrationResult, compute_curvature_hybrid, extract_topo_features, topo_adjusted_rho
- `exp_phase2_pipeline/test_zone_classifier.py` — валидация v1/v2/v3 на 35 графах
- `exp_phase2_pipeline/bench_preruntime.py` — бенчмарк pre-runtime overhead

**Возвращаемый TopoFeatures содержит:**
- Per-node: curvature (hybrid), pagerank, clustering_coeff, local_density, degree
- Per-cluster: mean/std/max агрегаты + boundary curvature
- Profiling: sigma_F, eta_F, gini_pagerank, **topo_zone** (GREEN/YELLOW/RED)

---

## Фаза 3 — Tree Semantics + Rebuild (22 марта 2026)

### Цель

Ответить на два вопроса: (1) Семантично ли дерево? (LCA-distance коррелирует с feature similarity?) (2) Работает ли инкрементальный rebuild? (anchors удерживают divergence < 5%?)

### Эксперименты и результаты

| Exp | Название | Конфигов | Kill criteria | Результат | Вердикт |
|-----|----------|----------|--------------|-----------|---------|
| exp14 | Anchors + Periodic Rebuild | 720 | divergence < 5% | scalar_grid/vector_grid: div=0.000 **PASS**. irregular_graph/tree_hierarchy: div 0.20-1.72 **FAIL** | **Partial FAIL** |
| exp15 | LCA-distance correlation | 80 | Spearman r > 0.3 | scalar_grid: 0.299 (borderline). vector_grid: -0.032. graph: 0.267. tree: 0.006 | **ALL FAIL** |
| exp15b | Bushes (leaf-path clusters) | 80 | Silhouette > 0.4 AND ARI > 0.6 | Silhouette: 0.49-0.79 **PASS**. ARI: -0.01 to 0.21 **FAIL** | **FAIL** |
| exp16 | C-pre (trajectory profiles) | 80 | Gap > 1.0 AND Sil > 0.3 | All 4 spaces PASS. Gap 1.37-2.40. k_mode 2-7. | **PASS → Track C UNFREEZE** |

### Ключевые находки

**Exp14 (anchors):** Local update идеально работает для регулярных пространств (grid: divergence = 0). Для нерегулярных (graph, tree) — structural drift. Dirty-triggered rebuild лучше periodic, но даже aggressive dirty_0.05 не проходит kill criteria. **Вывод:** проблема не в стратегии rebuild, а в том что монолитная ρ не разделяет структуру и задачу → мотивация для Phase 3.5.

**Exp15 (LCA-distance):** Дерево НЕ семантично в смысле LCA. Близость в дереве ≠ близость в feature space. scalar_grid ближе всего (0.299), но это артефакт регулярной сетки.

**Exp15b (bushes):** Кластеры leaf-paths **реальны** (Silhouette > 0.4 во всех пространствах), но **нестабильны** между seeds (ARI < 0.21). Потенциально: leaf-path similarity может использоваться для merge candidates, обнаружения дублирующих регионов, или как фичи для downstream. **Revisit запланирован после Track C.**

**Exp16 (C-pre):** Trajectory profiles образуют 2-7 natural clusters с Gap > 1.0 и Silhouette > 0.3 во ВСЕХ 4 пространствах. Pipeline ведёт себя дискретно — есть "типы поведения" при уточнении. **Track C разморожен.**

### Gate Phase 3 → Phase 3.5

- "Дерево семантично?" → **НЕТ** (P3a LCA FAIL + P3b bushes FAIL)
- "C unfreezes?" → **ДА** (C-pre PASS)
- "Rebuild работает?" → **Частично** (grid YES, graph/tree NO → мотивация для декомпозиции ρ)

### Ключевые файлы Phase 3

- `experiments/exp14_anchors/` — anchors + rebuild, 720 configs
- `experiments/exp15_lca_semantics/` — LCA-distance correlation, 80 configs
- `experiments/exp15b_bushes/` — leaf-path clustering, 80 configs
- `experiments/exp16_cpre_profiles/` — trajectory profile clustering, 80 configs

---

## Фаза 3.5 — Three-Layer Rho Decomposition (23 марта 2026)

### Предыстория

Phase 3 показала: монолитная ρ смешивает три concern'а — структуру пространства (topology), наличие данных (presence), и задаче-специфический запрос (query). Из-за этого: дерево нельзя переиспользовать, rebuild дрейфует на нерегулярных пространствах, непонятно что за что отвечает.

### Архитектура: три каскадных слоя

```
Layer 0: ТОПОЛОГИЯ        "как устроено пространство"
         Кластеры, мосты, хабы, плотность, границы
         [не зависит от данных вообще]

Layer 1: ДАННЫЕ ДА/НЕТ    "где в этом пространстве что-то лежит"
         Бинарная карта присутствия поверх топологии
         [не зависит от задачи]

Layer 2: ЗАПРОС            "из того что лежит — где нужное мне"
         residual / HF / teacher / что угодно
         [зависит от конкретной задачи]
```

Каждый слой **сужает** область работы для следующего. Переиспользуемость растёт снизу вверх.

### Cascade Quotas (Variant C)

**Проблема:** фиксированный L1 threshold (0.01) убивал 97% юнитов на scalar_grid при масштабе 1000. Reusability FAIL (0.725).

**Решение:** адаптивный порог L1, привязанный к кластерной структуре L0. Каждый L0-кластер гарантирует минимум выживших: `quota = max(1, ceil(cluster_size × min_survival_ratio))`. Информация течёт строго сверху вниз: L0 → L1 → L2. Ни один L0-кластер не вымирает полностью.

**Результат:** scalar_grid 1000: 0.725 FAIL → 0.863 PASS. Reusability 12/12 PASS по всем пространствам и масштабам.

### Streaming Pipeline

Вместо batch L0(all)→L1(all)→L2(all) — обработка по кластерам с L0-priority ordering и глобальным budget cap:

```
Cluster 0 (highest L0 score): [L0] → [L1] → [L2 refine]
Cluster 1:                           [L0] → [L1] → [L2 refine]
Cluster 2:                                  [L0] → [L1] → [L2 refine]
```

Два преимущества: (1) первые результаты после первого кластера; (2) L1 pruning реально сокращает refinement (budget per-cluster).

### Результаты (1080 конфигов, 0 ошибок)

| Метрика | Результат |
|---------|-----------|
| Reusability | 12/12 PASS (min ratio 0.838, threshold 0.80) |
| PSNR vs single_pass | Grids: -2-4 dB (цена L1 pruning). Graph/tree: паритет. |
| Streaming vs batch | 10-20% быстрее на grids |
| vs industry kdtree | kdtree быстрее на single query (оптимизированный C). 3L выигрывает при ≥2 запросах на tree_hierarchy. |
| Amortized break-even | tree_hierarchy: 2 запроса. Grids: не окупается (refinement доминирует). |

### Industry Baselines (сравнение)

| Baseline | Метод | Применимость |
|----------|-------|-------------|
| cKDTree (scipy) | k-d дерево + NN query | Все 4 пространства |
| Quadtree | Деление на квадранты по rho | Grid only |
| Leiden + brute force | Community detection + sort by rho | Graph only |
| Wavelets (Haar DWT) | Detail coefficients как saliency | Scalar grid only |

### C/Cython Roadmap

Бутылочное горлышко — refinement (numpy, ~60% total time). Scoring фазы (L0 topo, L1 presence, L2 query) — Python, ускоряемы 10× через C/Cython. Прогноз: streaming с C-scoring обойдёт kdtree на irregular_graph (70ms→5ms topo extraction) и сравняется на grids. Записано в workplan.md секция H.

### Ключевые файлы Phase 3.5

- `experiments/exp17_three_layer_rho/layers.py` — Layer0_Topology, Layer1_Presence, Layer2_Query, FrozenTree, ThreeLayerPipeline, IndustryBaselines
- `experiments/exp17_three_layer_rho/exp17_three_layer_rho.py` — runner с --chunk support
- `experiments/exp17_three_layer_rho/config17.py` — конфигурация эксперимента
- `experiments/exp17_three_layer_rho/results/` — результаты по чанкам (JSON)
- `experiments/exp17_three_layer_rho/README.md` — полное описание (RU+EN)

---

## Что дальше — Фаза 4

### P4a: Downstream Consumer Test
- Задача: классификатор или автоэнкодер на adaptive-refined данных vs dense vs coarse
- Kill criteria: metric loss < 2%
- Зависимости: все P0-P3.5

### P4b: Matryoshka
- Каждый уровень вложенности дерева — валиден для downstream
- Зависимости: P4a

### Bushes Revisit (после Track C)
- Кластеры leaf-paths реальны (Silhouette > 0.4), но нестабильны (ARI < 0.21)
- Потенциал: leaf-path similarity для merge candidates, обнаружение дублирующих регионов
- Идея: на основе leaf-path features — уплотнять кластеры, находить сходство между регионами

### C-Optimization (Roadmap)
- C/Cython переписка scoring фаз → 10× ускорение L0/L1/L2
- Streaming + C-scoring → потенциальный выигрыш над kdtree

### Бюджетный контроль — три ортогональных механизма

**ВАЖНО для следующей сессии:** в проекте есть путаница с термином "Governor". Вот точная картина:

**1. L1 Cascade Quotas** — "где есть данные" (hardware-invariant, structural)
- Каждый L0-кластер гарантирует минимум выживших юнитов
- Не зависит от железа — чисто структурный фильтр

**2. Budget Governor (hardware param + EMA feedback)** — "сколько обработать" (hardware-adaptive, dynamic)
- **Два слоя:** (a) hardware parameter задаёт ДИАПАЗОН (поводок) — мощное железо → широкий диапазон, слабое → узкий; (b) EMA feedback двигается ВНУТРИ диапазона на основе runtime-сигналов (waste rate, rejection rate, cost/step)
- **История:** в exp0.8 EMA governor работал (halved StdCost, cut P95 from 11→6.5). При сборке Phase 2 pipeline был **потерян** — GovernorIsolation в pipeline.py получает константу 1.0 и ни на что не влияет. StrictnessTracker + WasteBudget заменили его как бюджетный контроллер, но это ДРУГОЙ механизм (аварийный, не плавный).
- **Статус:** нужно восстановить в Phase 4. Hardware calibration уже есть (Synthetic Transport Probe, 52ms at startup).
- **GovernorIsolation** из exp10d — это НЕ бюджетный контроллер. Это EMA-телеметрия для DET-1 (проверка order-independence). Не путать!

**Область применения по режимам pipeline:**
| Режим | Governor EMA | Обоснование |
|-------|-------------|-------------|
| Batch (L0→L1→L2 раздельно) | ✅ Нужен | Feedback между полными шагами L2 |
| Frozen tree reuse (повторные L2) | ✅ Нужен | Адаптация между повторными L2 queries |
| Streaming (cluster-by-cluster) | ❌ Не нужен / вреден | Cross-cluster bleed: кластеры гетерогенны, feedback от RED-зоны не должен ужесточать GREEN |

В streaming бюджет контролируется global budget cap + per-cluster WasteBudget. EMA добавил бы шум (cross-cluster bleed).

**Открытый вопрос: плавное управление бюджетом в streaming.** Сейчас в streaming только бинарные механизмы ("go" / "stop"), нет плавной регулировки. Планируемое решение — комбинация двух подходов:
- **(B) L0-informed budget allocation (формула институционального неравенства):** L0 уже знает zone (GREEN/YELLOW/RED) каждого кластера. Вес кластера в бюджете:

  $$W_{cluster} = N_{units} \times (1 - ECR)^{\gamma}, \quad \gamma \geq 2$$

  **Вывод из термодинамики StrictnessTracker.** Ожидаемый drift strictness за шаг:

  $$E[\Delta S] = (1 - ECR) \times 0.9 + ECR \times 1.5$$

  - GREEN (ECR=0.05): E[ΔS] = 0.93 → strictness падает → кластер процветает
  - YELLOW (ECR=0.15): E[ΔS] = 0.975 → около равновесия
  - RED (ECR=0.33): E[ΔS] = 1.098 → strictness растёт → кластер умирает от WasteBudget

  Линейная аллокация (1-ECR) тратит бюджет впустую: RED получает 67% номинала, но не может их освоить (WasteBudget убьёт раньше). Квадратичная (γ=2) соответствует реальной пропускной способности:
  - GREEN: ~90% номинальной квоты
  - YELLOW: ~72%
  - RED: ~42%

- **(C) Adaptive budget redistribution:** если кластер N не израсходовал квоту — остаток перетекает к следующим (forward carry). RED получает строгий минимум, но при аномально чистом прогоне получает остатки от GREEN.

Нужно оттестировать в sweep вместе с Governor EMA (batch/reuse). γ sweep: γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} (от линейного baseline до агрессивного).

**3. WasteBudget + StrictnessTracker** — "аварийный стоп" (safety, per-unit memory)
- StrictnessTracker: per-unit множитель, escalation ×1.5 при reject, decay ×0.9 per clean step
- WasteBudget: R_max = floor(B_step × ω), каждый reject стоит strictness_multiplier единиц (не 1.0!), force-stop при waste ≥ R_max
- Это "самозатягивающаяся удавка" — радиоактивный хаб после 3 reject'ов стоит ~3.4 единицы waste, лавинообразно выбивая предохранитель
- **Работает сейчас** в pipeline.py (строки 421-439, 476)

### Exp18: Basin Membership (FAIL, deferred)

- Гипотеза: дерево = RG-flow, правильная метрика семантичности = basin membership (бассейн аттрактора), а не LCA-distance
- Результат: point-biserial r = 0.019, kill r > 0.3: **ALL FAIL**
- Причина: при 30% бюджете в single-pass бассейны не формируются (юниты не доходят до fixed points)
- **Deferred:** вернуться после multi-pass (Phase 4+), когда дерево будет достаточно глубоким

### MultiStageDedup (Phase 4)

- Код реализован (3 уровня: exact hash, metric distance, policy rule) в enox_infra.py
- Никогда не тестировался (`enox_dedup_enabled=False` по дефолту)
- Требует multi-pass / iterative refinement для осмысленной работы
- Запланирован как S4 в Phase 4

---

## Что было сделано ранее (Фаза 0)

### Эксперименты
1. **Окружение**: PC 1 (AMD Radeon 780M, DirectML) + PC 2 (RTX 2070, CUDA 12.8)
2. **Halo cross-space**: grid/graph OK, tree FAIL (0.56x). Правило: parallelism >= 3 AND no leakage.
3. **P2a sweep**: код готов (20K конфигураций), НЕ ЗАПУЩЕН
4. **SC-baseline**: D_parent = ||R(delta)|| / (||delta|| + eps), R = gauss sigma=3.0. AUC 0.824-1.000 на 4 пространствах.

### Ключевые архитектурные решения
- Halo: НЕ универсальный инвариант — правило по топологии.
- D_parent: формула обновлена (lf_frac).
- Morton/block-sparse/phase schedule: ОТЛОЖЕНЫ (deferred), не отвергнуты.

---

## Окружение

```bash
# Активация venv (PC 2):
R:\Projects\Curiosity\.venv-gpu\Scripts\activate
# Python 3.12.11, PyTorch 2.10.0+cu128, CUDA 12.8

# Git auth:
gh auth setup-git
# Токен: R:\Projects\.gh_tkn
```

## Принципы

- **Кросс-пространственная валидация** — 4 типа пространств обязательно
- **Kill criteria до запуска** — каждый эксперимент
- **Holm-Bonferroni** — при множественных сравнениях
- **10-20 seeds** — для воспроизводимости
- **Barmagloth = архитектор** — принимает решения на развилках, не пишет код
