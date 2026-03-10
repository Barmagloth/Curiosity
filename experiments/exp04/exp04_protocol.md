# Exp0.4 — Combined Interest: протокол казни или оправдания

## Цель

Ответить на вопрос: **обязательна ли комбинированная ρ, или достаточно одного сигнала?**

Ответ — в цифрах, не в философии.

---

## Варианты ρ

| # | Название | Формула | Назначение |
|---|----------|---------|------------|
| 1 | HF-only | `ρ = HF(coarse)` | Структурная энергия |
| 2 | Residual-only | `ρ = abs(upsample(coarse) - target)` | Ошибка приближения |
| 3 | HF + Residual | `ρ = w1·norm(HF) + w2·norm(Resid)` | Минимальная комбинация |
| 4 | HF + Residual + Variance | `ρ = w1·norm(HF) + w2·norm(Resid) + w3·norm(Var)` | Полная комбинация |
| 5 | **Anti-control** | `ρ = norm(HF) + norm(permute(HF))` | Проверка пайплайна |

**Anti-control (вариант 5)**: noise = перемешанные значения norm(HF). Сохраняет распределение, убивает структуру. Если "улучшает" метрики — протокол дырявый, эксперимент невалиден.

---

## Определения сигналов

Фиксируются **до запуска**, не меняются.

- **HF**: high-frequency энергия на **coarse** уровне (Laplacian). Вход: coarse (низкое разрешение).
- **Residual**: **всегда в target-grid**.
  ```
  coarse_up = upsample(coarse, target_shape, mode=bilinear)
  resid = abs(coarse_up - target)
  ```
  Затем агрегируется до tile-level через mean по тайлу.
  Никогда не считается на coarse-grid напрямую.
- **Variance**: `local_var(coarse, window=3x3)` — локальная дисперсия coarse.
- **Noise (anti-control)**: `permute(quantile_normalize(HF))` — перемешанные нормализованные значения HF. Та же нормализация, те же квантили, уничтоженная структура.

---

## Нормализация

Каждая компонента приводится к сопоставимой шкале:

```
z_i = clip((x_i - q50) / (q90 - q10 + eps), -3, 3)
```

- Квантили считаются **один раз** по всему полю на текущем уровне
- eps = 1e-8
- Веса w_i = 1/N_components (равные), или depth-dependent (задаётся в конфиге)
- Anti-control noise проходит ту же нормализацию, что и остальные компоненты

---

## Gain: определение и тип

**gain = marginal** — улучшение MSE от применения данного тайла в текущем состоянии (greedy order).

```
gain_marginal[tile] = MSE_before_tile - MSE_after_tile
```

Для Exp0.4: порядок фиксирован (top-K по ρ, все применяются одновременно, gain = MSE тайла в coarse vs target).

В per-split логе пишется явно: `gain_type = marginal`.

**gain_oracle** (изолированный gain тайла без влияния соседей) — опционален. Если реализован, логируется отдельно, но решения принимаются по gain_marginal.

---

## Пороги: δ_false и δ_miss (разведены)

### δ_false — порог ложного сплита

```
δ_false = 0.05 * median(gain_marginal)
```

- Считается по **HF-only, seeds 0..2**
- Фиксируется **до основных прогонов**, далее не трогается
- False split = тайл выбран, но gain_marginal < δ_false

### δ_miss — порог пропущенной структуры

```
δ_miss = q75(gain_marginal) по dense-прогону (budget=100%)
```

- Dense-прогон: все тайлы включены, считается gain каждого
- Ground truth "важный тайл" = gain_marginal > δ_miss в dense-прогоне
- Фиксируется **один раз** per seed

---

## Метрики

### Primary endpoints (решают)
- **MSE** при фиксированном бюджете
- **false_split_rate** = доля выбранных тайлов с gain < δ_false

Статтест: paired Wilcoxon по seed'ам + **Holm-Bonferroni** поправка (2 primary × 4 сравнения с baseline).

### Secondary endpoints (report, не решают)

#### Miss rate (две категории)

| Тип | Определение | Диагноз |
|-----|-------------|---------|
| `miss_detector` | GT-тайл не вошёл в top по ρ (score < median(ρ)) | Лечится ρ |
| `miss_budget` | GT-тайл был в top по ρ, но не вошёл из-за budget cap | Лечится schedule/бюджетом |

GT-тайл определяется через δ_miss.

#### Stability

1. **Input-jitter**: `σ = 0.005 * (max - min)`
2. **Transform-jitter**: subpixel shift (0.3, 0.3) px

Метрики: `IoU(mask_base, mask_perturbed)`, `Δnodes`

#### ROI
- `median(ROI)`, `q10(ROI)`, `ROI_per_level`

Вторичные метрики — **без поправки на множественные сравнения**. Это явно отмечено.

---

## Логирование

### Per-split

```
level, tile_id, ρ_total, ρ_HF, ρ_resid, ρ_var, gain_marginal, gain_type, split_decision
```

### Per-run

```
variant, seed, budget, PSNR, MSE,
false_split_rate, miss_detector, miss_budget,
stability_IoU_input, stability_IoU_transform, Δnodes_input, Δnodes_transform,
ROI_median, ROI_q10
```

### Корреляции

```
corr(ρ_HF, gain), corr(ρ_resid, gain), corr(ρ_var, gain), corr(ρ_total, gain)
```

---

## Протокол запуска

1. Dense-прогон → δ_miss per seed
2. Calibration: HF-only seeds 0..2 → δ_false (фиксация)
3. Запуск всех вариантов × all seeds
4. Метрики
5. Stability: два jitter-протокола
6. Статтесты: paired Wilcoxon + Holm-Bonferroni на primary

---

## Критерий принятия

**Принимается**, если:
1. MSE не хуже лучшего одиночного (p > 0.05, Holm-Bonferroni)
2. И false_split_rate значимо лучше (p < 0.05, Holm-Bonferroni)
3. И ни одна вторичная не деградировала > 20%

**Отклоняется**, если:
- Нет значимого улучшения ни по одному primary
- Или anti-control "тоже улучшает" → протокол невалиден

---

## Anti-self-deception checklist

- [ ] Residual в target-grid (upsample → abs → tile-mean)
- [ ] Gain = marginal, тип в логе
- [ ] δ_false и δ_miss разведены, источники зафиксированы
- [ ] δ_false зафиксирована до основных прогонов
- [ ] HF/Residual/Variance — определения не менялись
- [ ] Anti-control = permute(norm(HF)), та же нормализация
- [ ] Нормализация одинакова для всех вариантов
- [ ] Anti-control запущен и проанализирован
- [ ] Корреляции ρ_i с gain залогированы
- [ ] Miss rate разделён на detector/budget
- [ ] Stability с конкретными σ / shift
- [ ] ≥ 10 seed'ов
- [ ] Paired Wilcoxon + Holm-Bonferroni на primary
- [ ] Вторичные без поправки — явно отмечено
