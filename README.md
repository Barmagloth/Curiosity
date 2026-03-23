# Curiosity — Adaptive Refinement System

> **[English version / Английская версия](README_ENG.md)**

## Что это

Curiosity — исследовательский ML-проект по созданию системы **адаптивного уточнения** (adaptive refinement) для абстрактных вычислительных пространств.

Система не привязана к изображениям или конкретной предметной области. Она работает с произвольным пространством состояний X, где функция информативности ρ(x) определяет, где уточнение оправдано.

## Ключевая идея

> Размерность — это не фиксированное число, а глубина уточнения.
> Уточнение должно происходить только там, где это оправдано информационно и бюджетно.

Цель — изменить пространство поиска так, чтобы:
- его можно было исследовать **адаптивно**
- вычислительный ресурс тратился только там, где это даёт информацию
- при этом сохранялись взаимосвязи признаков (не «ломались фичи»)

## Статус проекта

Серия экспериментов Exp0.1–Exp0.8 **завершена**. Результаты консолидированы в документацию v1.8. Система валидирована: адаптивное уточнение работает и превосходит random selection при ограниченном бюджете. В v1.6 добавлен (обновлён в v1.7) Scale-Consistency Invariant — формализация требования «не сломать фичи». В v1.8 добавлены инварианты детерминизма (DET-1, DET-2).

Phase 0 (Exp0.1–Exp0.8) завершена. Phase 1 (P0 layout, DET-1, sensitivity, scale-consistency) завершена. P0 Layout **закрыт** — финальная policy по всем типам пространств зафиксирована в `docs/layout_selection_policy.md`. Серия exp10 (8 субэкспериментов, 158 000+ trials) определила оптимальный layout для каждого типа пространства. Phase 2 (end-to-end pipeline validation) завершена. Phase 3 (anchors, LCA-distance, bushes, C-pre) завершена — Track C UNFREEZE. Phase 3.5 (three-layer rho decomposition, exp17) завершена — архитектурная декомпозиция ρ на L0/L1/L2, reusability 12/12 PASS.

Следующий рубеж — **Phase 4** (P4a: downstream consumer test, P4b: matryoshka).

## Структура репозитория

```
README.md                          — этот файл
README_ENG.md                      — English version
docs/                              — документация (русский)
  concept_v1.8.md                  — концептуальный документ (канонический)
  layout_selection_policy.md       — методика подбора layout (P0 результат)
  experiment_hierarchy.md          — граф зависимостей экспериментов и roadmap
  session_handoff.md               — документ передачи между сессиями
  phase1_plan.md                   — план Phase 1 (завершена)
  target_problem_definition_v1.1.md — зачем проект, что успех, Track A→B→C
  scale_consistency_verification_protocol_v1.0.md — протокол верификации SC
  architecture.md                  — архитектура системы
  workplan.md                      — план реализации (модули A–F)
  glossary.md                      — глоссарий терминов проекта
  teamplan.md                      — план для команды
  handoff.md                       — документ передачи проекта (legacy)
  experiment_results.md            — результаты Exp0.1–Exp0.8
  concept_v1.7_historical.md       — историческая версия (после Phase 0)
  concept_v1.6_historical.md       — историческая версия (после Phase 0)
  concept_v1.5_historical.md       — историческая версия (после Exp0.1–0.8)
  concept_v1.4_historical.md       — ранняя версия (после Exp0.2–0.3)
  handoff_v1.5_to_v1.6.md          — changelog v1.5→v1.6
  environment_1.md                 — окружение PC 1
  environment_2.md                 — окружение PC 2 (RTX 2070, CUDA)
docs_eng/                          — documentation (English)
  [зеркальная структура docs/]
experiments/
  exp01_poc/                       — Exp0.1: PoC adaptive refinement
  exp02_cifar_poc/                 — Exp0.2: CIFAR PoC
  exp03_halo_diagnostic/           — Exp0.3: halo diagnostic
  exp04_combined_interest/         — Exp0.4: combined interest
  exp05_break_oracle/              — Exp0.5: break oracle
  exp06_adaptive_switch/           — Exp0.6: adaptive ρ switch
  exp07_gate/                      — Exp0.7/0.7b: soft gate + two-stage
  exp08_schedule/                  — Exp0.8: dynamic schedule + governor + probe
  exp09a_layout_sandbox/           — Exp0.9a: layout microbench (CPU sandbox)
  exp10_buffer_scaling/            — P0: grid vs compact на GPU (серия exp10)
  exp10d_seed_determinism/         — DET-1: побитовый детерминизм
  exp10e_tile_sparse/              — P0: tile-sparse кандидаты (A/B/C)
  exp10f_packed_lookup/            — P0: packed tiles + direct/hash lookup
  exp10g_dual_benchmark/           — P0: dual-mode benchmark (stencil + conv2d)
  exp10h_cross_space/              — P0: cross-space (vector_grid + tree)
  exp10i_graph_blocks/             — P0: блочная адресация для графов
  exp10j_tree_perlevel/            — P0: per-level break-even для деревьев
  exp10k_cost_surface/             — P0: cost-surface analysis
  exp11_dirty_signatures/          — dirty signature compression
  exp11a_det2_stability/           — DET-2: cross-seed stability
  exp12a_tau_parent/               — data-driven τ_parent по глубине
  exp13_segment_compression/       — сжатие сегментов дерева
  exp14_anchors/                   — Phase 3: anchors + periodic rebuild
  exp14a_sc_enforce/               — SC-enforce: three-tier pass/damp/reject
  exp15_lca_semantics/             — Phase 3: LCA-distance vs feature similarity
  exp15b_bushes/                   — Phase 3: leaf-path clustering (bushes)
  exp16_cpre_profiles/             — Phase 3: C-pre trajectory profiles → Track C UNFREEZE
  exp17_three_layer_rho/           — Phase 3.5: three-layer rho (L0/L1/L2)
  exp_phase2_pipeline/             — Phase 2: full pipeline assembly
  exp_phase2_e2e/                  — Phase 2: end-to-end validation
  exp_deferred_revisit/            — research note: отложенные вопросы
  halo_crossspace/                 — halo applicability across space types
  p2a_sensitivity/                 — sensitivity sweep порогов гейта
  phase1_halo/                     — Phase 1: halo/overlap hardening
  phase2_probe_seam/               — Phase 2: probe + seam metric
  sc_baseline/                     — SC-baseline: scale-consistency
```

## Рекомендованный порядок чтения

Для нового участника проекта:

1. **`docs/target_problem_definition_v1.1.md`** — зачем проект, что считается успехом
2. **`docs/concept_v1.8.md`** — каноническая концепция (все валидированные решения)
3. **`docs/glossary.md`** — термины проекта
4. **`docs/architecture.md`** — архитектура и компоненты
5. **`docs/layout_selection_policy.md`** — методика подбора layout по типам пространств
6. **`docs/experiment_results.md`** — результаты Exp0.1–Exp0.8 с числами
7. **`docs/experiment_hierarchy.md`** — граф зависимостей и roadmap
8. **`docs/handoff.md`** — документ передачи (статус + первая задача)
9. **`docs/workplan.md`** — план реализации

## Технологии

- Python, PyTorch, CUDA (GPU)
- Jupyter Notebooks для экспериментов
- Versioned markdown для документации

## Самостоятельно ценные компоненты

Каждый модуль проектируется так, чтобы **выживать поодиночке**, даже если «большое дерево» не взлетит:

- **Content-addressable cache** — каноникализация + контентный хэш тайлов, переиспользуемый кэш для любого pipeline.
- **Инкрементальный пересчёт** — пересчёт только изменившихся регионов, общая оптимизация.
- **ROI-маска «интересности»** — функция ρ как самостоятельный инструмент адаптивного внимания.
- **Адаптивная разметка сложности** — quadtree/octree + гистерезис как независимая карта сложности данных.
- **SeamScore** — метрика качества швов, применимая к любому тайловому pipeline.

Принцип отбора: выживают только модули, которые дают выигрыш сами по себе.

## Методология

- Каждое утверждение подкреплено данными ("judge numbers, then ambitions")
- Forensic-grade протоколы: явные контроли, Holm-Bonferroni коррекции, observable-only диагностики
- Cost-fair сравнения: вычислительные затраты учтены
- Kill criteria перед запуском каждого эксперимента
- Никакой oracle-информации в метриках

## Авторство

Проект ведёт Barmagloth. Исследование проводилось с использованием Claude (Anthropic) и ChatGPT (OpenAI) как ассистентов.
