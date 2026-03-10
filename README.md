# Curiosity — Adaptive Refinement System

## Что это

Curiosity — исследовательский ML-проект по созданию системы **адаптивного уточнения** (adaptive refinement) для абстрактных вычислительных пространств.

Система не привязана к изображениям или конкретной предметной области. Она работает с произвольным пространством состояний X, где функция информативности p(x) определяет, где уточнение оправдано.

## Ключевая идея

> Размерность — это не фиксированное число, а глубина уточнения.
> Уточнение должно происходить только там, где это оправдано информационно и бюджетно.

Цель — изменить пространство поиска так, чтобы:
- его можно было исследовать **адаптивно**
- вычислительный ресурс тратился только там, где это даёт информацию
- при этом сохранялись взаимосвязи признаков (не «ломались фичи»)

## Статус проекта

Серия экспериментов Exp0.1–Exp0.8 **завершена**. Результаты консолидированы в документацию v1.5. Система валидирована: адаптивное уточнение работает и превосходит random selection при ограниченном бюджете.

Следующий рубеж — Exp0.9b0 (buffer-scaling probe, P0).

## Структура репозитория

```
README.md                          — этот файл
docs/
  concept_v1.5.md                  — концептуальный документ (канонический)
  concept_v1.4_historical.md       — ранняя версия концепции (после Exp0.2–0.3)
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

## Технологии

- Python, PyTorch, CUDA (GPU)
- Jupyter Notebooks для экспериментов
- Versioned markdown для документации

## Методология

- Каждое утверждение подкреплено данными ("judge numbers, then ambitions")
- Forensic-grade протоколы: явные контроли, Holm-Bonferroni коррекции, observable-only диагностики
- Cost-fair сравнения: вычислительные затраты учтены
- Kill criteria перед запуском каждого эксперимента
- Никакой oracle-информации в метриках

## Авторство

Проект ведёт Barmagloth. Исследование проводилось с использованием Claude (Anthropic) и ChatGPT (OpenAI) как ассистентов.
