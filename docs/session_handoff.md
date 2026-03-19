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

## Что делать — Фаза 1

**Детальный план:** `docs/phase1_plan.md`

Запустить параллельных воркеров:
- **Worker A** (GPU): S1 exp10_buffer_scaling → S1b exp10d_seed_determinism
- **Worker B** (CPU): S2 exp11_dirty_signatures
- **Worker C** (CPU): S3 запуск p2a_sensitivity (код готов!)
- **Worker D** (CPU): S4 exp12a_tau_parent
- **Worker E**: S5 deferred revisit (низший приоритет)

### Критический путь
```
P0 (exp10) → DET-1 (exp10d) → Phase 2
```

### Развилки для архитектора (конец Фазы 1)
- P0: grid или compact?
- P2a: ridge width — manual ok или нужен adaptive? Одинаков между пространствами?
- SC-5: pass/fail?
- Deferred: возвращать Morton/block-sparse/schedule?

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
