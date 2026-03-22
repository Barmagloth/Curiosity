# Session Handoff — Curiosity Phase 2

Документ для новой сессии AI-оркестратора. Содержит полный контекст для немедленного продолжения работы.

## Где мы

Проект Curiosity.

- Фаза 0 **завершена** (18 марта 2026).
- Фаза 1 **завершена** (20 марта 2026). Все потоки — PASS. P0 Layout **ЗАКРЫТ**. DET-1 **PASS**. DET-2 **PASS**.
- Фаза 2 **завершена** (20 марта 2026). Pipeline assembled, SC-enforce integrated, E2E validated.
- **Enox-инфраструктура** — ✅ DONE (21 марта 2026). Четыре observation-only паттерна (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep) — чистая аннотация, не меняют pipeline state.
- **Следующий шаг — Фаза 3.**

Рабочий ПК: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Рабочая директория: `R:\Projects\Curiosity`.

## Что читать (в этом порядке)

| # | Файл | Зачем |
|---|------|-------|
| 1 | `docs/session_handoff.md` | **Этот файл** — текущий статус, что делать дальше |
| 2 | `docs/concept_v1.8.md` | Каноническая концепция (актуальная) |
| 3 | `docs/teamplan.md` | План с отметками Фаза 0, Фаза 1, описание Фаз 2-4 |
| 4 | `docs/experiment_hierarchy.md` | Граф зависимостей, приоритеты, нумерация exp10+ |
| 5 | `docs/environment_2.md` | Как активировать .venv-gpu на PC 2 (CUDA) |
| 6 | `docs/phase1_plan.md` | Архив — план и результаты Фазы 1 (справочно) |

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

1. **Governor гистерезис для irregular high budget** — Track B. Как настроить EMA-контроллер для пространств с нерегулярным бюджетным профилем.

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
