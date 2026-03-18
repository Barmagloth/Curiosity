# Session Handoff — Curiosity Phase 1

Документ для новой сессии AI-оркестратора. Содержит полный контекст для немедленного продолжения работы.

## Где мы

Проект Curiosity. Фаза 0 (параллельная валидация + настройка окружения) **завершена**. Все результаты закоммичены и запушены в main.

## Что было сделано в этой сессии

### Документация
- Исправлены broken references в handoff.md (concept_v1.5→v1.6→v1.7)
- Добавлен рекомендованный порядок чтения в README
- Создан teamplan.md (RU + ENG) — план параллельной работы команды
- Добавлен .gitignore

### Фаза 0 — эксперименты и код
1. **Окружение**: .venv (CPU, Python 3.13) + .venv-gpu (DirectML, Python 3.12, AMD Radeon 780M)
2. **Halo cross-space**: Halo работает на grid/graph, FAIL на tree (0.56×)
   - Правило: boundary parallelism ≥ 3 AND no context leakage
   - Код: experiments/halo_crossspace/
3. **P2a sweep**: код готов (20K конфигураций), НЕ ЗАПУЩЕН
   - Код: experiments/p2a_sensitivity/
4. **SC-baseline**: D_hf pass, D_parent pass (после фиксов)
   - Финальная формула: D_parent = ‖R(δ)‖ / (‖δ‖ + ε), R = gauss σ=3.0
   - Cross-space: T1=1.000, T2=1.000, T3=1.000, T4=0.824
   - coarse_shift генератор исправлен (coherent sign fields)
   - Код: experiments/sc_baseline/

### Ключевые архитектурные решения
- Halo: НЕ универсальный инвариант. Применяется только при parallelism≥3 + no leakage.
- D_parent: формула изменена. Старая (α·‖coarse‖+β в знаменателе) — нулевой дискриминативный сигнал.
- Morton/block-sparse/phase schedule: ОТЛОЖЕНЫ (deferred), не отвергнуты.
- Кросс-пространственная валидация: обязательна для любых утверждений об "arbitrary spaces" (4 типа: scalar grid, vector grid, irregular graph, tree hierarchy).

## Что делать дальше — Фаза 1

По плану в docs/teamplan.md, раздел "Фаза 1". Barmagloth — архитектор, исполнители — AI-агенты.

### Потоки Фазы 1 (параллельные):
1. **P0: Exp0.9b0** — Buffer-scaling probe на GPU (DirectML). Grid vs compact. Kill compact если overhead >20%. Использовать .venv-gpu.
2. **P1-B2**: Dirty signatures прототип (CPU). 12-bit signature, debounce, AUC>0.8.
3. **P2a**: ЗАПУСК sensitivity sweep (код уже готов в experiments/p2a_sensitivity/). 5 сцен × 4 пространства.
4. **SC-baseline завершение**: SC-5 — установить data-driven τ_parent[L]. Подготовить SC-enforce.
5. **Deferred revisit**: Research note по Morton/block-sparse/phase schedule с новыми подходами.

### Развилки для архитектора (конец Фазы 1):
- P0: grid или compact?
- P2a: ridge width — manual ok или нужен adaptive?
- P2a cross-space: ridge одинаков или различается между пространствами?
- Deferred: возвращать Morton/block-sparse/schedule?

## Ключевые файлы для ориентации

| Файл | Зачем читать |
|------|-------------|
| docs/concept_v1.8.md | Каноническая концепция (актуальная) |
| docs/teamplan.md | План работы команды с отметками о выполнении |
| docs/experiment_hierarchy.md | Граф зависимостей, приоритеты, roadmap |
| docs/experiment_results.md | Все числа экспериментов |
| docs/environment_1.md | Как активировать окружение |
| experiments/halo_crossspace/results/APPLICABILITY_RULE.md | Правило применимости Halo |
| experiments/sc_baseline/results/CROSSSPACE_SC_REPORT.md | Финальные результаты SC |

## Принципы

- **Кросс-пространственная валидация** — обязательна (4 типа пространств)
- **Kill criteria до запуска** — каждый эксперимент
- **Holm-Bonferroni** — при множественных сравнениях
- **10–20 seeds** — для воспроизводимости
- **Git author**: Barmagloth <>
