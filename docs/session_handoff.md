# Session Handoff — Curiosity Phase 1

Документ для новой сессии AI-оркестратора. Содержит полный контекст для немедленного продолжения работы.

## Где мы

Проект Curiosity. Фаза 0 **завершена** (18 марта 2026). Фаза 1 **спланирована, не начата**.

Рабочий ПК: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Рабочая директория: `R:\Projects\Curiosity`.

## Что читать (в этом порядке)

| # | Файл | Зачем |
|---|------|-------|
| 1 | `docs/phase1_plan.md` | **План Фазы 1** — 6 потоков, worker assignment, kill criteria, reuse map |
| 2 | `docs/concept_v1.8.md` | Каноническая концепция (актуальная) |
| 3 | `docs/experiment_hierarchy.md` | Граф зависимостей, приоритеты, нумерация exp10+ |
| 4 | `docs/teamplan.md` | План с отметкой Фаза 0 ✅, описание Фаз 1-4 |
| 5 | `docs/environment_2.md` | Как активировать .venv-gpu на PC 2 (CUDA) |

## Что было сделано ранее (Фаза 0)

### Эксперименты
1. **Окружение**: PC 1 (AMD Radeon 780M, DirectML) + PC 2 (RTX 2070, CUDA 12.8)
2. **Halo cross-space**: grid/graph OK, tree FAIL (0.56×). Правило: parallelism ≥ 3 AND no leakage.
3. **P2a sweep**: код готов (20K конфигураций), НЕ ЗАПУЩЕН
4. **SC-baseline**: D_parent = ‖R(δ)‖ / (‖δ‖ + ε), R = gauss σ=3.0. AUC 0.824–1.000 на 4 пространствах.

### Ключевые архитектурные решения
- Halo: НЕ универсальный инвариант — правило по топологии.
- D_parent: формула обновлена (lf_frac).
- Morton/block-sparse/phase schedule: ОТЛОЖЕНЫ (deferred), не отвергнуты.

## Фаза 1 — Результаты (19 марта 2026)

| Stream | Результат |
|--------|-----------|
| S1 exp10 | KILL compact-with-reverse-map (VRAM +38.6%). Grid — baseline. Но убита реализация, не принцип sparse. |
| S1b exp10d | DET-1 PASS (240/240 побитовое совпадение CPU+CUDA) |
| S2 exp11 | FAIL 3/4 spaces. Архитектурная проблема. |
| S3 P2a | PASS — ridge 100%. Ручные пороги ok. P2b не нужен. |
| S4 exp12a | Thresholds найдены. L1 specificity низкая. |
| S5 deferred | Research note done. |

Gate Phase 1→2: PASSED (grid fixed + DET-1). P0 переоткрыт для layout investigation.

## Фаза 1b — exp10e результаты (19 марта 2026)

| Candidate | Time vs grid | VRAM vs grid | Verdict |
|-----------|-------------|--------------|---------|
| A (bitset) | **-27% to -31%** | +18% median | **ALIVE** — execution layout, не memory layout |
| B (packed Morton) | +825% to +1503% | -30% (sparse) to +243% | **KILLED** — binary search lookup. Storage idea жива. |
| C (paged) | +5352% to +9769% | mixed | **KILLED** — окончательно |

## Что делать — Фаза 1c (exp10g)

exp10f результат: D passes Contour A, fails Contour B (peak VRAM from conv2d workspace). E archived as contingency.

Текущий статус кандидатов:
- **A** = operational default (grid+bitset, быстрее grid на 27-31%)
- **D** = passes Contour A, pending Contour B resolution via exp10g
- **E** = archived fallback with resurrection triggers

**exp10g — dual-mode benchmark:** manual stencil (layout cost) vs conv2d (operator cost).
Цель: разделить layout cost и operator cost, resolves D's Contour B failure.

Kill criteria: те же (>20% overhead vs grid per pattern class).

### Критический путь
```
exp10g (dual benchmark) → layout decision → Phase 2
```

### Развилки для архитектора (конец Фазы 1c)
- exp10g: manual stencil vs conv2d — что доминирует в VRAM?
- D rehabilitated or killed?
- exp11: переделывать signatures или принять ограничение?
- exp12a: L1 без enforcement или доработка?

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
- **10–20 seeds** — для воспроизводимости
- **Barmagloth = архитектор** — принимает решения на развилках, не пишет код
