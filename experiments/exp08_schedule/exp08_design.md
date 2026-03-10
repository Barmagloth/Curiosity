# Exp0.8 — Нужен ли schedule? (финальный дизайн)

## Два этапа

### Этап 1: Governor (A vs B)
- rho одинаковая (combined, фиксированные веса)
- A: strictness = фиксированный квантиль (калиброванный на шаге 0)
- B: strictness двигается EMA-контроллером
- Вопрос: EMA лучше по compliance + качеству при равном total cost?

### Этап 2: Policy (B vs C)
- Governor одинаковый (EMA)
- B: rho weights фиксированные
- C: rho weights = phase schedule (линейный, замороженный)
- Вопрос: phase schedule даёт выигрыш сверх governor?

## Механика выбора тайлов

1. Вычисляем rho для всех доступных тайлов (rank-normalized per step)
2. Порог = quantile(rho_available, strictness)
3. exploit = все тайлы выше порога, отсортированные desc
4. hard_cap: не больше 3 × target_per_step за шаг (safety fuse)
5. probe: PROBE_FRAC от target_per_step, из тайлов ниже порога
6. Фактический расход = len(exploit) + len(probe)
7. Каждый тайл стоит 1 единицу
8. Episode заканчивается когда total_cost >= TOTAL_BUDGET

## Бюджет

- TOTAL_BUDGET = 80 (на эпизод)
- TARGET_PER_STEP = TOTAL_BUDGET / N_STEPS = 4
- HARD_CAP_PER_STEP = 3 × TARGET = 12
- Коридор: [target × 0.7, target × 1.3]
- Probe: 10% от target = 1 тайл/шаг минимум

## Anti-oscillation

- Δstrictness за шаг ≤ CLAMP = 0.03
- EMA α = 0.3
- Warmup: первые 3 шага — strictness не двигается (но cost считается)
  Метрики compliance считаются с шага warmup+1

## Compliance (асимметричная)

```
penalty_step = max(0, cost - upper) × 2.0    # overbudget: вес 2
             + max(0, lower - cost) × 1.0    # underbudget: вес 1
```

Итоговый compliance = mean(penalty_step) по шагам (после warmup).

## Холодный старт

strictness_init: на шаге 0 подбираем квантиль, при котором count
ближайший к target_per_step. Оба варианта (A и B) стартуют с него.

## Phase schedule (C) — заморожен

```
w_resid = 0.7 − 0.4 × (step / (N_STEPS−1))   # 0.7 → 0.3
w_var   = 0.15 + 0.2 × (step / (N_STEPS−1))   # 0.15 → 0.35
w_hf    = 0.15 + 0.2 × (step / (N_STEPS−1))   # 0.15 → 0.35
```

Эти числа зафиксированы ДО прогона. Не тюнятся.

## Probe discovery

- err_true(tile) = MSE(gt_tile, output_tile) — фактическая ошибка
  (НЕ совпадает с rho, которая = комбинация resid+var+hf)
- discovery = probe-тайл с err_true в top-25%
- probe_contribution: ablation — прогон с probe applied vs probe logged-only,
  при том же seed. Разница PSNR = вклад probe. (делаем counterfactual на эпизод)

## Повторяемость

- N_SEEDS = 20
- Отчёт: median + IQR
- Verdict по распределению (Wilcoxon signed-rank), не по одному числу

## Safety fuse

- hard_cap_per_step = 3 × target
- В логах: cap_triggered_count

## Метрики

1. **Quality**: PSNR на момент исчерпания бюджета
2. **Compliance**: asymmetric penalty (mean, после warmup)
3. **Stability**: cost_variance, cost_p95 (после warmup)
4. **Discovery**: probe discovery rate, probe contribution (ablation)
5. **Efficiency**: PSNR / total_cost (quality per unit cost)
6. **Cap triggers**: сколько раз hard_cap сработал

## Сцены

- clean, noise, spatvar (3 основных)
- shift (GT меняется на полпути)

## Verdict

### Этап 1 (Governor):
B лучше A, если при равном total cost (±5%):
- compliance penalty ниже (median, p < 0.05 Wilcoxon)
- cost_variance ниже
- PSNR не хуже (или лучше)

### Этап 2 (Policy):
C лучше B, если при одинаковом governor:
- PSNR выше (median, p < 0.05)
- не за счёт перерасхода
