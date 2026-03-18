# Curiosity — Adaptive Refinement System

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

Серия экспериментов Exp0.1–Exp0.8 **завершена**. Результаты консолидированы в документацию v1.6. Система валидирована: адаптивное уточнение работает и превосходит random selection при ограниченном бюджете. В v1.6 добавлен Scale-Consistency Invariant — формализация требования «не сломать фичи».

Следующий рубеж — Exp0.9b0 (buffer-scaling probe, P0) и SC-baseline (верификация scale-consistency).

## Структура репозитория

```
README.md                          — этот файл
docs/
  target_problem_definition_v1.1.md — зачем проект, что успех, Track A→B→C
  concept_v1.6.md                  — концептуальный документ (канонический)
  concept_v1.5_historical.md       — предыдущая версия (после Exp0.1–0.8)
  concept_v1.4_historical.md       — ранняя версия (после Exp0.2–0.3)
  scale_consistency_verification_protocol_v1.0.md — протокол верификации SC
  handoff_v1.5_to_v1.6.md          — changelog v1.5→v1.6
  experiment_results.md            — результаты всех экспериментов Exp0.1–Exp0.8
  experiment_hierarchy.md          — граф зависимостей экспериментов и roadmap
  architecture.md                  — архитектура системы и ключевые решения
  workplan.md                      — план реализации (модули A–F, мини-роадмап)
  handoff.md                       — документ передачи проекта
  glossary.md                      — глоссарий терминов проекта
experiments/
  ARTIFACT_INVENTORY.md            — инвентарь артефактов из диалогов Claude/ChatGPT
  exp01_poc/                       — Exp0.1: PoC adaptive refinement (IPYNB)
  exp02_cifar_poc/                 — Exp0.2: CIFAR PoC (IPYNB)
  exp03_halo_diagnostic/           — Exp0.3: halo diagnostic (IPYNB)
  exp04_combined_interest/         — Exp0.4: combined interest (код+протокол+данные)
  exp05_break_oracle/              — Exp0.5: break oracle (код+данные)
  exp06_adaptive_switch/           — Exp0.6: adaptive ρ switch (код)
  exp07_gate/                      — Exp0.7/0.7b: soft gate + two-stage (код+протокол+данные)
  exp08_schedule/                  — Exp0.8: dynamic schedule (код+дизайн+данные)
  exp09a_layout_sandbox/           — Exp0.9a: layout microbench (план §C / C3)
  phase1_halo/                     — План валидации §A (A1+A2+A3): halo/overlap hardening
  phase2_probe_seam/               — План валидации §B (B1+B2): probe + seam metric
```

## Рекомендованный порядок чтения

Для нового участника проекта:

1. **`docs/target_problem_definition_v1.1.md`** — зачем проект, что считается успехом
2. **`docs/concept_v1.6.md`** — каноническая концепция (все валидированные решения)
3. **`docs/glossary.md`** — термины проекта
4. **`docs/architecture.md`** — архитектура и компоненты
5. **`docs/experiment_results.md`** — результаты Exp0.1–Exp0.8 с числами
6. **`docs/experiment_hierarchy.md`** — граф зависимостей и roadmap
7. **`docs/handoff.md`** — документ передачи (статус + первая задача)
8. **`docs/workplan.md`** — план реализации

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
