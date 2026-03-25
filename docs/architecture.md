# Архитектура Curiosity (Phase 4, 25 марта 2026)

Документ описывает текущую архитектуру системы по состоянию на Phase 4 (multi-tick pipeline).
Терминология соответствует `docs/glossary.md`.

---

## 1. Обзор системы

Curiosity --- adaptive refinement для абстрактных пространств. Система уточняет
представление только там, где функция информативности ρ показывает, что это оправдано
информационно и бюджетно. Размерность --- не фиксированное число осей, а глубина
уточнения.

Поддерживаемые типы пространств:

| Тип | Описание |
|-----|----------|
| scalar_grid | 2D скалярная сетка (image-like) |
| vector_grid | 2D векторная сетка (multi-channel) |
| irregular_graph | Нерегулярный граф (k-NN, Swiss Roll и др.) |
| tree_hierarchy | Иерархия деревьев (branching factor варьируется) |

Ключевая формула уточнения: `refined = parent_coarse + step_delta`.
Step_delta инициализируется нулём, ограничена по энергии; при отключении текущего
уровня система откатывается к parent_coarse.

---

## 2. Pipeline архитектура (Phase 4, multi-tick)

Pipeline реализован в `experiments/exp_phase2_pipeline/pipeline.py`.

**Outer loop (multi-tick):** каждый tick пересчитывает unit_rho (L2 query), FSR,
instability, затем выполняет inner loop. Количество тиков: `max_ticks` (config).

**Inner loop (per tick):**
1. Canonical sort юнитов по rho (Z-order tie-break, DET-1)
2. Per-unit: compression guard (tree) --> dedup check (Enox) --> refine_unit() --> SC-enforce --> ROI check --> track
3. Probe allocation из неоценённых/неуточнённых юнитов
4. Governor EMA update

**Backward compat:** `max_ticks=1` (default) воспроизводит Phase 2 поведение побитово.
В single-tick mode пропускаются: cold-start, FSR/instability, ROI check, convergence
detector, weighted rho. DET-1 recheck: 40/40 PASS.

**Tick budget:** `target_per_tick = n_budget / max_ticks`. При исчерпании ---
`budget_exhausted` stop.

---

## 3. WeightedRhoGate

Замена TwoStageGate (Phase 2). Вместо дискретного переключения Stage 1/Stage 2 ---
единая функция:

```
ρ = Σ(w_i × signal_i),  i ∈ {resid, hf, var}
```

EMA-веса (alpha = `ema_weight_alpha`, default 0.3) плавно переходят между состояниями:
- resid здоров (instability <= thresh AND FSR <= thresh): w_resid -> 1.0
- resid нестабилен: w_resid снижается пропорционально excess instability, HF/var получают
  60%/40% от остатка. Минимальный w_resid = `resid_min_weight` (0.20).

**Cold-start:** начальные пороги instability/FSR вычисляются из:
- Topo zone (GREEN=0.15, YELLOW=0.25, RED=0.40 --- prior на instability)
- CV(initial_rho) --- эмпирическая оценка нестабильности
- Blend: 50% prior + 50% empirical, умноженный на `pilot_thresh_factor`

**Pilot fine-tuning:** первые `pilot_ticks` (default 3) тиков собирают реальные
instability/FSR; на последнем pilot-тике пороги уточняются: min(cold-start, median
observed * pilot_thresh_factor).

**SC-enforce инвариантен к весам:** delta = refine_unit() не зависит от ρ. ρ определяет
КТО уточняется, не КАК. Смена весов не ломает scale-consistency.

---

## 4. Трёхслойная декомпозиция ρ (L0 / L1 / L2)

Монолитная ρ декомпозируется на три каскадных слоя (exp17, Phase 3.5):

| Слой | Назначение | Частота пересчёта |
|------|-----------|-------------------|
| L0 (Topology) | Структура пространства: кластеры, мосты, кривизна | Один раз при init |
| L1 (Presence) | Где есть данные: variance GT per unit | Каждый step (инкрементально) |
| L2 (Query) | Задаче-специфическое уточнение: residual, HF, max abs error | Каждый query |

Каждый слой сужает рабочее множество для следующего. Переиспользуемость растёт
снизу вверх.

**L0 реализация:** irregular_graph --- Leiden communities + curvature + PageRank;
scalar/vector_grid --- spatial quadrant blocks; tree_hierarchy --- depth-band grouping.

**L1 cascade quotas (Variant C):** каждый L0-кластер гарантирует
`max(1, ceil(cluster_size * budget_fraction))` выживших юнитов. Ни один регион
не вымирает.

**Frozen Tree:** сериализуемый снапшот L0+L1 (скоры + active units). Аналог R-tree.
Строится один раз, разные L2-запросы работают поверх.

**Streaming pipeline:** покластерная обработка (L0->L1->L2 per cluster),
L0-priority ordering, глобальный budget cap. Первые результаты --- после первого
кластера. 10-20% быстрее batch.

---

## 5. Бюджетный контроль (3 механизма)

Три ортогональных механизма:

**1. L1 cascade quotas (структурный).** Встроены в трёхслойную ρ. Топология диктует
бюджет по кластерам. Streaming формула аллокации:
`W_cluster = N_units * (1 - ECR)^gamma`, gamma >= 2. GREEN кластеры получают ~90%
номинала, RED ~42%. Forward carry перераспределяет остатки.

**2. Governor EMA (hardware-adaptive).** EMA(cost) -> strictness corridor.
`governor_ema_enabled=False` по умолчанию. Двухслойная архитектура: hardware range +
EMA feedback. Параметры: corridor_hi=1.5, corridor_lo=0.5, strictness_clamp=0.05,
warmup_ticks=3. Работает в batch/reuse mode, НЕ в streaming.

**3. WasteBudget + StrictnessTracker (safety).** Самозатягивающаяся петля:
- StrictnessTracker: per-unit множители, эскалация x1.5 при reject, затухание x0.9
- WasteBudget: R_max = floor(B_step * omega), reject стоит strictness_multiplier единиц
- Force-stop при waste >= R_max
- Параметры: waste_omega=0.2, strictness_escalation=1.5, strictness_decay=0.9

---

## 6. SC-enforce

Scale-Consistency Invariant: step_delta не должна переопределять семантику
parent_coarse. Формально: `D_parent = ||R(step_delta)|| / (||step_delta|| + eps) < tau_parent`.

R = gaussian blur (sigma=3.0) + decimation. Up = bilinear upsampling. Пара фиксирована.

**Three-tier enforcement:**
- `D_parent <= tau_parent` --> **pass** (delta принимается as-is)
- `D_parent > tau_parent` после damping --> **damp** (delta *= damp_factor, до max_damp_iterations=3)
- Damping не помогло --> **reject** (state откатывается)

**Adaptive tau для tree_hierarchy:** tau_parent[L, space_type] задаётся per-space
(youden_j из baseline). Grids ~0.42-0.50, graph ~0.08, tree ~0.19.

**Topo zone modifiers (irregular_graph):** tau_eff = tau_parent * zone_factor.
GREEN=1.3 (relax), YELLOW=1.0, RED=0.7 (tighten).

D_hf = ||delta - Up(R(delta))|| / (||delta|| + eps) --- диагностика, не hard constraint.

---

## 7. Halo + Probe

### Halo (boundary-aware blending)

Cosine feathering на границах тайлов, предотвращает ложные HF-сигналы на швах.

| Топология | Halo? | Параметр |
|-----------|-------|----------|
| scalar_grid, vector_grid | Да | halo_width=2 (grid spaces) |
| irregular_graph | Да | halo_hops=1 |
| tree_hierarchy | Нет | boundary parallelism < 3, context leakage |

Правило применимости: boundary parallelism >= 3 AND no context leakage.

### Probe (exploration)

Бюджет: `probe_fraction` (default 10%) от tick budget.

Phase 4: probe выбирается только из юнитов, НЕ входящих в refined_set и
evaluated_set (Issue 7). DeterministicProbe: seed = f(coords, level, global_seed).

Probe --- страховка от ложных fixed points и структурной слепоты.

---

## 8. Topo profiling (irregular_graph)

Многоступенчатый анализ топологии, выполняется при инициализации ДО первого tick.

**Hybrid curvature engine:**
1. Forman-Ricci для ВСЕХ рёбер (O(1) per edge)
2. Сортировка по |Forman| аномалии
3. Upgrade top-N к Ollivier-Ricci (EMD, O(W^3)), N = floor(topo_budget / t_ollivier)

**Synthetic Transport Probe:** аппаратная калибровка (~52ms), определяет kappa_max.

**Признаки:** sigma_F (std Forman), eta_F (entropy index = sigma_F / sqrt(2 * mean_degree)),
Gini(PageRank).

**Zone classification (v3):**
- kappa_mean > 0 --> GREEN (плотные клики, ECR < 5%)
- kappa < 0 AND Gini < 0.12 AND eta_F <= 0.70 --> YELLOW (регулярные решётки)
- Иначе --> RED (структурный хаос)

Валидация: 34/35 на 35-graph корпусе. eta_F порог 0.70 из мёртвой зоны [0.60, 0.76].

---

## 9. Enox infrastructure

Четыре observation-only паттерна. Все default OFF (`enox_*_enabled=False`).
Не модифицируют pipeline state.

| Паттерн | Назначение |
|---------|-----------|
| **RegionURI** | SHA256-based стабильный адрес юнита: `SHA256(parent_id \| op_type \| child_idx)` -> 16 hex |
| **DecisionJournal** | Append-only лог решений gate/SC-enforce с метриками (URI, action, rho, D_parent) |
| **MultiStageDedup** | 3-уровневая дедупликация: exact hash -> metric (epsilon) -> policy. epsilon=0.0 default |
| **PostStepSweep** | Поиск merge candidates: sibling-юниты с overlap > `enox_sweep_threshold` (0.05) |

MultiStageDedup --- заготовка для multi-pass; в single-pass (epsilon=0.0) не срабатывает.

---

## 10. Compression guard (tree_hierarchy)

**Thermodynamic guards:** segment compression для degree-2 transit nodes. Критерий:
N_critical = 12 (N_CRITICAL_D2). Degree-2 nodes с стабильным содержимым пропускаются
в inner loop (d2_skip_set).

**Динамическая деактивация:** внутри inner loop проверяется should_compress() ---
если guards перестают выполняться (n_active снизился, budget исчерпан), d2_skip_set
очищается и compression прекращается.

Применяется только к tree_hierarchy.

---

## 11. Layout policy

Выбор layout --- функция типа пространства (статически известен):

| Пространство | Layout | Обоснование |
|-------------|--------|-------------|
| scalar_grid | D_direct (packed tiles + tile_map) | exp10g: both contours PASS |
| vector_grid | D_direct | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid per-level: D_direct при p_l < 0.40 AND heavy compute; A_bitset иначе | exp10j: 158K trials |
| irregular_graph (spatial) | D_blocked (cbr <= 0.35) | exp10i |
| irregular_graph (scale-free) | A_bitset (fallback) | cbr=0.66, blocked rejected |

D_direct: активные тайлы в компактном массиве, tile_map[tile_id] -> slot (int32).
A_bitset: полноразмерный тензор + bitset mask. D_blocked: узлы в блоках фиксированного
размера, block_map[block_id] -> slot.

Layout Selection Invariant (гипотеза, Track C): C(I, M, p) --- отложена, не доказана.

---

## 12. Convergence + ROI

**Convergence detector (Issue 1):** если accepted == 0 за `convergence_window` (default 2)
последовательных тиков --> stop с reason "zero_accepted". Только в multi-tick mode.

**ROI gating (Issue 6):** после tick 0 каждый юнит проверяется:
`roi = unit_rho(before) - unit_rho(after)`. Если roi < reference_median_gain *
min_roi_fraction (0.15) --> skip. Reference_median_gain берётся из tick 0.

**ROI fix (exp19):** global MSE -> local unit_rho reduction. Причина: изменение одного
юнита даёт пренебрежимое изменение global MSE на больших пространствах, что приводит
к ложным reject. После fix: mt=3 восстанавливает 99% от single-tick PSNR (было 37%).

---

## 13. Детерминизм

**DET-1 (Seed determinism, Hard Constraint):** одни данные + ρ + seed + бюджет =
побитово идентичное дерево. Три компонента:

1. **Canonical traversal:** Z-order/Morton index tie-break при равном ρ
2. **Deterministic probe:** seed = f(coords, level, global_seed)
3. **Governor isolation:** EMA-update строго после полного шага, canonical order

DET-1 recheck Phase 4: 40/40 PASS.

**DET-2 (Cross-seed stability, Soft Constraint):** CV < tau_cv для PSNR, cost,
compliance, SeamScore по разным seeds. Phase 1: PASS.

---

## 14. Модули A-H (из workplan)

| Модуль | Назначение | Статус |
|--------|-----------|--------|
| A | Каноникализация + контентный хэш тайлов | НЕ реализован (не на критическом пути) |
| B | Планировщик пересчёта по изменению | НЕ реализован (не на критическом пути) |
| C | Интересность ρ как измеряемая функция | Валидирован (Exp0.4-0.7, двухстадийный гейт) |
| D | Дерево + split/merge | Валидирован (Exp0.1-0.3, Exp0.8 governor) |
| E | Дельты + границы (halo) | Валидирован (halo w=[2,4], SeamScore, Phase 1/2) |
| F | Бенчмарк + стоимость управления | Проведён (exp10, 158K+ trials, layout policy) |
| G | Scale-Consistency | Валидирован (SC-baseline AUC 0.82-1.0, exp12a tau_parent PASS) |
| H | Трёхслойная декомпозиция ρ | Валидирован (exp17, 1080 конфигов, reusability 12/12 PASS) |

---

## 15. Статус валидации

### Фазы

| Фаза | Дата | Результат |
|------|------|-----------|
| Phase 0 | 18.03.2026 | Environment setup, halo cross-space validation, SC-baseline |
| Phase 1 | 20.03.2026 | P0 Layout ЗАКРЫТ. DET-1 PASS. DET-2 PASS |
| Phase 2 | 20.03.2026 | E2E pipeline assembled, SC-enforce integrated, topo profiling |
| Enox | 21.03.2026 | 4 observation-only паттерна |
| Phase 3 | 22.03.2026 | Exp14 anchors (grid PASS, graph/tree FAIL). Exp15/15b FAIL. Exp16 C-pre PASS -> Track C UNFREEZE |
| Phase 3.5 | 23.03.2026 | Three-layer rho. 1080 конфигов, reusability 12/12 PASS |
| Exp18 | 23.03.2026 | Basin membership FAIL (r=0.019). Deferred post-multi-pass |
| Phase 4 | 25.03.2026 | Multi-tick pipeline. WeightedRhoGate. Issues 1-7 resolved. DET-1 40/40 PASS |

### Exp19 (multi-tick sweep, 2050 конфигов, 0 ошибок)

| Sub-exp | Конфигов | Результат |
|---------|----------|-----------|
| 19a scaling law | 840 | mt=2-3 оптимум. vector_grid: +6-19% |
| 19b gate stress | 160 | 160/160 PASS. alpha=0.3 оптимален |
| 19c param sweep | 420 | Чистая синтетика: multi-tick = overhead |
| 19d real data | 150 | CIFAR mt=3: 96-97%. Real graphs: 100%. Overhead <5% |
| 19e noisy+hetero | 480 | Шум: +2-7%. Чистый hetero: -26-30%. Mixed sigma>=0.10: +4-7% |

**Ключевой вывод:** multi-tick полезен при шуме (gate адаптирует w_resid 1.0->0.84).
На чистых данных single-tick оптимален.

### Открытые вопросы

- **Issue 8:** post-refinement quality feedback (Phase 4+)
- **Issue 9:** noise-fitting --- refinement копирует шумный GT (Phase 5)
- **Phase 5:** P5-noise (exp20) --- denoising refinement, 6 подходов x 3 sigma x 10 seeds
- **Track C:** C-оптимизация scoring (roadmap)
- **P4a:** downstream consumer test
- **P4b:** matryoshka invariant validation
- **Bushes:** revisit после Track C
- **RG-flow basins:** revisit после multi-pass

---

## Стек технологий

- **Язык:** Python (CPU-only core logic, no PyTorch dependency)
- **ML-фреймворк:** PyTorch (для экспериментов)
- **GPU:** CUDA (PC 2, RTX 2070) / DirectML (PC 1, AMD)
- **Кластеризация:** Leiden (igraph/leidenalg, primary), Louvain (NetworkX, fallback)
- **Эксперименты:** Jupyter Notebooks + Python scripts
- **Документация:** Versioned markdown
- **Конфигурация:** `experiments/exp_phase2_pipeline/config.py` (dataclass PipelineConfig)
