# Phase 2 — Итоговый отчёт

## Содержание

1. Probe-budget exploration (B1+B2)
2. Seam metric v1 → v2 (развитие локальной метрики швов)
3. Cross-space validation (grid, vector, граф, дерево)
4. Зафиксированные правила для concept
5. Открытые вопросы

---

## 1. Probe-budget exploration

### Конфигурация

- Grid: 128×128, tile=4, 1024 тайлов
- Budget: 6%/step (61 тайл), probe=10% бюджета (6 тайлов), 25 шагов
- Overlap: w=3, cosine feather, cost = (tile + 2w)²
- Governor: выключен (равные условия)
- Seeds: 7

### Сцены

**TP**: Громкий фон (ступеньки + синусоиды, residual до 0.094) + тихая текстура в 4 островках (66 тайлов, residual 0.006). Quiet tiles НЕ попадают в exploitation top-61.

**TN**: Тот же фон, тихие регионы = шум (amp=0.0005).

### Результаты probe-стратегий

#### TP (структура скрыта, probe должен найти)

| Стратегия | Global PSNR | Quiet PSNR | Discover rate | Discover step | False act | ROI |
|-----------|-------------|------------|---------------|---------------|-----------|-----|
| no_probe  | 37.85 ± 0.02 | 38.41 | 0% | — | 0 | 0 |
| uniform   | 37.94 ± 0.02 | 38.39 | 100% | step 0.9 | 35 | 7.8e-6 |
| age       | 37.85 ± 0.02 | 37.88 | 0% | — | 39 | 3.7e-6 |
| uncert    | 37.86 ± 0.01 | 38.37 | 100% | step 7.4 | 7 | 2.8e-5 |
| bandit    | 37.88 ± 0.03 | 37.90 | 0% | — | 36 | 4.2e-6 |

#### TN (структуры нет, probe тратит впустую)

| Стратегия | Global PSNR | Quiet PSNR | Discover rate | False act | ROI |
|-----------|-------------|------------|---------------|-----------|-----|
| no_probe  | 37.86 ± 0.00 | 37.45 | 0% | 0 | 0 |
| uniform   | 37.98 ± 0.03 | 37.78 | 100% | 36 | 7.8e-6 |
| age       | 37.88 ± 0.00 | 37.38 | 0% | 36 | 3.5e-6 |
| uncert    | 37.86 ± 0.00 | 37.48 | 100% | 9 | 2.7e-5 |
| bandit    | 37.92 ± 0.00 | 37.38 | 0% | 36 | 4.1e-6 |

### Выводы по probe

1. **Probe нужен**: no_probe discover = 0%, exploitation структурно слепа к тихим регионам.
2. **PSNR-вклад probe мал** (±0.1 dB) — ожидаемо. Probe = страховка от слепоты, не оптимизация.
3. **uncert — лучшая стратегия**: 100% discovery, false activation 7 (vs 35–39 у остальных), ROI в 4× выше.
4. **uniform — fallback**: обнаруживает всё, но шумит.
5. **age/bandit не работают** при 25 шагах: TTL не успевает, bandit не прогревается.
6. **TN discover = 100% для uniform/uncert** — артефакт: при исчерпании exploitation любой свежий тайл конкурирует по gain. Нужна привязка к абсолютному порогу окупаемости.

---

## 2. Seam metric: v1 → v2

### v1 (concentric rings)

**Конструкция**: SeamScore = Jump_out / (Jump_in + eps). Rings вокруг tile boundary, Chebyshev distance.

**Результаты v1 (S1+S2, w=0..5, signals: smooth/medium/sharp):**

| Signal | w=0 ΔSeam | w=1 | w=2 | w=3 | w=4 | w=5 |
|--------|-----------|-----|-----|-----|-----|-----|
| smooth | −7.3M | +0.75 | +0.10 | +0.02 | −0.91 | +0.02 |
| medium | +7.7M | +0.60 | +0.17 | +0.08 | −0.17 | +0.01 |
| sharp  | +87.6M | +0.22 | +0.02 | +0.03 | −0.07 | +0.01 |

**Проблемы v1:**
- w=0: деление на eps → SS = 10⁸
- w=4: ΔSeam отрицательный (ложное "улучшение") — кольца цепляют чужие тайлы
- Немонотонность по w — артефакт геометрии колец
- SS_before нестабилен (coarse-ступеньки попадают в кольца)

**S3 (dual check) v1 — работает:**

| Signal | Normal pass | Soap pass | n |
|--------|-------------|-----------|---|
| smooth | 77/81 | 25/81 | 81 |
| medium | 76/81 | 16/81 | 81 |
| sharp  | 70/81 | 4/81 | 81 |

### v2 (edge strips)

**Изменения**: убраны кольца → edge strips по halo boundary. Guard для w=0. Robust stat = median.

**Результаты v2 (S1+S2, w=0..3, valid range):**

| Signal/seed | w=0 ΔSeam | w=1 | w=2 | w=3 | Тренд |
|-------------|-----------|-----|-----|-----|-------|
| smooth/42   | −0.32 | +0.51 | +0.00 | +0.01 | ✓ plateau w≥2 |
| smooth/137  | −0.25 | +0.51 | −0.01 | −0.02 | ✓ стабильно |
| medium/42   | −0.47 | +0.28 | +0.15 | +0.15 | ✓ снижение |
| medium/137  | −0.47 | +0.29 | +0.14 | +0.13 | ✓ снижение |
| sharp/42    | −0.60 | +0.45 | +0.04 | +0.09 | ✓ снижение |
| sharp/137   | −0.63 | +0.44 | +0.02 | +0.04 | ✓ снижение |

**w ≥ TILE = дегенерация**: при w=4 для tile=4 SS взрывается (ΔSeam = −2.4 млрд). Причина: halo boundary уходит за соседний тайл. Правило: w < tile_size.

**S3 (dual check) v2:**

| Signal | Normal pass | Soap pass | n |
|--------|-------------|-----------|---|
| smooth | 80/81 | 25/81 | 81 |
| medium | 80/81 | 16/81 | 81 |
| sharp  | 80/81 | 4/81 | 81 |

**S4 (L2 vs L∞/p95) v2:**

| Signal | w | L2(median) | p95 | Ratio p95/L2 |
|--------|---|------------|-----|--------------|
| smooth | 2 | 0.131 | 0.034 | 0.26 |
| smooth | 3 | 0.243 | 0.036 | 0.15 |
| medium | 2 | 0.155 | 0.034 | 0.22 |
| medium | 3 | 0.175 | 0.019 | 0.11 |
| sharp  | 2 | 0.134 | 0.052 | 0.39 |
| sharp  | 3 | 0.272 | 0.054 | 0.20 |

L∞(p95) стабильнее L2 при малых w (L2 скачет из-за Jump_in нормализации, p95 — абсолютный).

### Seam Ratio (Phase 1 revision)

Универсальная метрика: SR = E[grad²_state] / E[grad²_GT] at seam pixels.

| Signal | w_opt | SR(w=0) | SR(w_opt) |
|--------|-------|---------|-----------|
| smooth | 2 | 3.6–4.0 | 1.41–1.46 |
| medium | 3 | 4.2–4.3 | 1.37–1.39 |
| sharp  | 4 | 3.6–4.5 | 1.24–1.34 |

Phase 1 "w=3 порог" уточняется: w_opt ∈ [2, 4] зависит от сигнала. Но SR требует GT → непригодна для реального латента.

---

## 3. Cross-space validation

### Dual check: normal vs soap pass rate

| Пространство | Halo=0 | Halo=1 | Halo=2 | Halo=3 |
|---|---|---|---|---|
| **T1: Scalar grid** (64×64, tile=8) | 6/6 vs 0/6 ✓ | 5/6 vs 0/6 ✓ | 5/6 vs 0/6 ✓ | 6/6 vs 0/6 ✓ |
| **T2: Vector grid** (64×64, tile=8, dim=32) | 6/6 vs 0/6 ✓ | 0/6 vs 0/6 ✗ | 6/6 vs 0/6 ✓ | 6/6 vs 0/6 ✓ |
| **T3: Irregular graph** (500pts, k=8) | 6/6 vs 0/6 ✓ | 3/6 vs 1/6 ✓ | 5/6 vs 1/6 ✓ | — |
| **T4: Tree** (depth=8, 255 nodes) | 4/4 vs 0/4 ✓ | 4/4 vs 2/4 ✓ | 3/4 vs 2/4 ~ | — |

Формат: normal_pass/n vs soap_pass/n.

### ΔSeam и gain по пространствам

| Space | Halo | ΔSeam normal | ΔSeam soap | Gain normal | Gain soap |
|-------|------|-------------|------------|-------------|-----------|
| Scalar grid | w=0 | +0.000 | +0.000 | 0.110 | 0.000 |
| Scalar grid | w=2 | +0.133 | +0.207 | 0.094 | 0.000 |
| Scalar grid | w=3 | +0.142 | +0.155 | 0.087 | 0.000 |
| Vector grid | w=0 | +0.000 | +0.000 | 0.015 | 0.000 |
| Vector grid | w=2 | +0.187 | +0.207 | 0.014 | 0.000 |
| Vector grid | w=3 | +0.137 | +0.155 | 0.013 | 0.000 |
| Graph | hop=1 | +0.497 | +0.156 | 0.106 | 0.001 |
| Graph | hop=2 | +0.181 | +0.174 | 0.100 | 0.001 |
| Tree | hop=0 | −0.169 | +0.000 | 0.075 | 0.000 |
| Tree | hop=1 | −0.033 | +0.230 | 0.075 | 0.002 |
| Tree | hop=2 | +0.012 | −0.115 | 0.075 | 0.002 |

### Ключевые наблюдения

1. **Dual check разделяет normal и soap во всех пространствах** при адекватном halo. Soap проходит dual check только когда halo слишком велик (tree hop=2) или tile слишком мал (vector w=1).

2. **Gain — главный разделитель.** Soap gain ≈ 0 везде (кроме tree hop≥1 где soap случайно аппроксимирует через усреднение). ΔSeam — вторичный фильтр, ловит случаи когда gain > 0 но шов разрушен.

3. **Vector grid, w=1: ΔSeam = 0.66** — cosine feather в 1 пиксель не успевает развернуть 32-мерный градиент. Все тайлы rejected. Минимум w=2 для high-d.

4. **Tree, hop=2: soap ΔSeam = −0.115** (отрицательный = "улучшение"). Soap на 2-hop halo в дереве = грубая аппроксимация, которая может сгладить шов. Dual check ослабевает потому что soap gain = 0.002 > threshold.

5. **Graph: soap leak = 1/6** при hop≥1. Причина: один кластер на дисконтинуити, soap случайно уменьшает MSE усреднением.

---

## 4. Зафиксированные правила для concept

### Probe (B1+B2)

1. Probe обязателен (10% бюджета).
2. Variance-based probe — лучшая стратегия (discovery 100%, false 7, ROI 2.8e-5).
3. Uniform — fallback.
4. Probe — страховка от слепоты, PSNR-вклад мал (ожидаемо).

### Seam metric

5. **SeamScore = Jump_out / (Jump_in + eps)** — универсальная локальная метрика шва.
6. Jump считается по edge strips (boundary pairs halo↔outside), не по кольцам.
7. Guard для w=0: SS = Jump_after / (Jump_before + eps).
8. **w < tile_size** — обязательное ограничение.
9. **Dual check: gain ≥ thr ∧ ΔSeam ≤ limit** — минимальный контроль качества refine.
10. Dual check работает на scalar grid, vector grid, irregular graph, tree hierarchy.
11. L∞(p95) — дополнительный канал для high-d (ловит "тонкий шов" в подпространстве).

### Halo (уточнение Phase 1)

12. w_opt ∈ [2, 4] зависит от типа сигнала (smooth→2, medium→3, sharp→4).
13. Phase 1 "w=3 порог" — не универсальный скачок, а минимум SR-кривой.
14. Для high-d (vector grid): w ≥ 2 обязательно (w=1 недостаточно).

---

## 5. Открытые вопросы

1. **TN false discovery.** Нужен абсолютный порог окупаемости для discovery, не relative ranking.
2. **Tree hop=2 soap leak.** Dual check ослабевает при большом halo в связных структурах. Нужен adaptive limit или gain threshold пропорциональный hop count.
3. **Auto-w через ΔSeam.** Pilot показал что ΔSeam minimum = w_opt, но нужна проверка на tile_size ≥ 8 где дискретизация достаточна.
4. **Random projections для high-d.** Реализованы L2/L∞, но random directions (8-16 фиксированных) не протестированы. Для реального dim=768 понадобится.
5. **Probe fraction adaptation.** 10% фиксировано. Следующий шаг: adaptive probe fraction через bandit на уровне fraction, не тайлов.

---

## Артефакты

### Probe exploration (B1+B2)

| Файл | Содержание |
|------|-----------|
| exp_phase2_probe.py | Эксперимент: 5 стратегий × 2 сцены × 7 seeds |
| phase2_probe.png | Global/Quiet PSNR vs step, smoke, summary bars |
| phase2_quiet_timeline.png | Quiet tiles refined per step (TP и TN) |
| phase2_probe.json | Сырые данные: все шаги всех прогонов |

### Seam metric v1 (concentric rings — deprecated)

| Файл | Содержание |
|------|-----------|
| exp_seam_metric.py | Эксперимент v1: кольца, S1–S5 |
| phase2_seam_metric.png | Графики v1 |
| phase2_seam_metric.json | Сырые данные v1 |

### Seam metric v2 (edge strips — актуальная)

| Файл | Содержание |
|------|-----------|
| exp_seam_metric_v2.py | Эксперимент v2: edge strips, S1–S5 |
| seam_metric_v2.png | ΔSeam vs w, SS vs w, dual check, L2 vs p95 |
| seam_metric_v2.json | Сырые данные v2 |

### Cross-space validation

| Файл | Содержание |
|------|-----------|
| exp_seam_crossspace.py | 4 типа пространств: scalar grid, vector grid, graph, tree |
| seam_crossspace.png | Dual check pass rates across spaces |
| seam_crossspace.json | Сырые данные cross-space |

### Отчёты

| Файл | Содержание |
|------|-----------|
| phase2_full_report.md | Этот документ (финальный) |
| phase2_summary_report.md | Ранний черновик (только probe, до seam metric) |
