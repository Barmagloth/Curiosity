# Scale-Consistency Verification Protocol (v1.0)

Протокол верификации Scale-Consistency Invariant для проекта Curiosity.

Этот документ — процедурный. Концептуальное обоснование — в `concept_v1.7.md`, раздел 8.

**Статус протокола (обновлено после Phase 0):**
- Шаги 0–4: ✅ **ЗАВЕРШЕНЫ**. SC-baseline пройден, D_parent валидирован.
- Шаг 5: 🔓 **ОТКРЫТ** — установка data-driven τ_parent[L].
- Шаг 6: 🔓 **ОТКРЫТ** — enforcement.

**Изменения по результатам Phase 0:**
- R: gaussian blur σ=1.0 → **σ=3.0** (σ=1.0 был недостаточно агрессивным LP-фильтром)
- D_parent: `‖R(δ)‖ / (α·‖coarse‖ + β)` → **`‖R(δ)‖ / (‖δ‖ + ε)`** (lf_frac — операционно корректная нормализация)
- Negative baseline: coarse_shift генератор исправлен — per-pixel sign flip → spatially coherent sign fields
- Валидация: пройдена на 4 типах пространств (scalar grid, vector grid, irregular graph, tree hierarchy)

---

## Контекст

Перед тем как вводить enforcement scale-consistency, необходимо убедиться что метрики D_parent и D_hf несут дискриминативную информацию. Этот протокол определяет как это проверить и когда остановиться.

---

## Шаг 0. Предварительная фиксация

До любых экспериментов зафиксировать и закоммитить:

| Параметр | Значение | Обоснование |
|---|---|---|
| R | gaussian blur **σ=3.0** + decimation | Линейный, GPU-дешёвый, согласован с coarse+delta. ~~σ=1.0~~ → σ=3.0 по результатам Phase 0 (σ=1.0 недостаточно подавлял positive delta, AUC=0.685→0.811) |
| Up | bilinear upsampling | Стабильный, интерпретируемый, без внесения чужей семантики |
| ε | small constant (1e-4 · mean(‖delta‖)) | Стабилизатор в D_parent и D_hf |

**~~Старые параметры (deprecated):~~**
| ~~α~~ | ~~1.0~~ | ~~Нормировка к coarse~~ | Убран — знаменатель заменён на ‖δ‖ |
| ~~β~~ | ~~1e-4·mean(‖coarse‖)~~ | ~~Защита от деления на ноль~~ | Заменён на ε |

**Пара (R, Up) не меняется в ходе одного цикла верификации.** Если нужно проверить другую пару — это отдельный цикл.

Проверить идемпотентность R: `‖R(coarse) − coarse_downsampled‖` должно быть пренебрежимо мало.

---

## Шаг 1. Подготовка данных

### 1.1 Positive baseline (корректные случаи)

Два уровня доверия — хранить раздельно:

**Strong positive** (приоритетный):
- Синтетические данные с known GT: `delta = GT − coarse` (near-oracle)
- Степень доверия высокая: scale-consistency гарантирована по построению

**Empirical positive** (вспомогательный):
- Случаи из Exp0.1–Exp0.8 с высоким PSNR и подтверждённым выигрышем
- Степень доверия ниже: хороший PSNR не гарантирует отсутствие semantic drift, только внешний результат
- Использовать для проверки: совпадают ли распределения D_parent/D_hf со strong positive? Если нет — интересный диагностический сигнал

Оба типа нужны; смешивать в одну выборку не следует до проверки совместимости распределений.

### 1.2 Negative baseline (намеренно плохие случаи)

Случаи, где инвариант намеренно нарушен:

- **LF drift:** к корректному delta добавить синусоиду низкой частоты (масштаб > tile_size)
- **Coarse shift:** delta намеренно смещает coarse-среднее региона на 10–30%. **NB:** генератор должен использовать spatially coherent sign fields, не per-pixel random sign flip (последний самогасится под R, создавая искусственно слабую violation — исправлено в Phase 0, см. `experiments/sc_baseline/baselines_v2.py`)
- **Random LF delta:** delta = случайный низкочастотный шум, не связанный с GT
- **Semant-wrong:** delta переворачивает знак coarse в регионе (экстремальный случай)

Для каждого negative case сохранять тип нарушения как метку.

---

## Шаг 2. Вычисление метрик

Для каждого (узел, уровень, тип) вычислить:

```python
# R = gaussian_blur(sigma=3.0) + downsample
# Up = bilinear_upsample_to_original_size

R_delta = R(delta)                        # coarse-проекция delta
P_LF_delta = Up(R_delta)                  # LF-компонента delta в исходном масштабе
delta_HF = delta - P_LF_delta             # HF-остаток

# Обновлённая формула (Phase 0, lf_frac normalization):
D_parent = norm(R_delta) / (norm(delta) + eps)   # "какая доля энергии delta — низкочастотная?"
D_hf     = norm(delta_HF) / (norm(delta) + eps)  # "какая доля энергии delta — высокочастотная?"

# DEPRECATED (v1.0 original):
# D_parent = norm(R_delta) / (alpha * norm(coarse) + beta)
# Причина замены: знаменатель α·‖coarse‖+β одинаков для всех delta на одном тайле →
# нулевой дискриминативный сигнал, CV=3.3 в распределении D_parent.
```

Сохранять: `(D_parent, D_hf, level, structure_type, case_type: pos/neg, neg_type)`

---

## Шаг 3. Анализ separability

**Ориентация метрик (зафиксировать до подсчёта):**

| Метрика | Направление | Смысл |
|---|---|---|
| D_parent | выше = хуже | negative cases должны давать бо́льшие значения |
| D_hf | выше = лучше | positive cases должны давать бо́льшие значения |

AUC считается с учётом ориентации. Для D_hf "positive = больший D_hf" означает, что при подсчёте ROC-AUC метка positive и negative назначаются соответственно. Если получить AUC < 0.5 без учёта ориентации — это не смерть метрики, а перепутанный знак.

### 3.1 Глобальная separability

Для каждой метрики (D_parent, D_hf):
- ROC-AUC: positive vs. negative
- PR-AUC
- Effect size (Cohen's d или rank-biserial correlation)
- Quantile separation (с учётом ориентации): D_parent: `median(neg) > q75(pos)`; D_hf: `median(pos) > q75(neg)`

### 3.2 Depth-conditioned separability

Повторить 3.1 отдельно для каждого уровня L.

Причина: метрика может хорошо работать на L=1 и разваливаться на L=3.

### 3.3 Regime-conditioned separability

Повторить 3.1 отдельно для каждого типа структуры (гладкое, граница, текстура).

---

## Шаг 4. Kill criterion (фиксируется до запуска)

### Acceptance threshold

Метрика считается архитектурно годной если **все** условия выполнены:

| Критерий | Порог |
|---|---|
| Global ROC-AUC | ≥ 0.75 |
| Depth-conditioned AUC (каждый уровень) | ≥ 0.65 |
| Effect size | ≥ medium (d ≥ 0.5) |
| Quantile separation | D_parent: `median(neg) > q75(pos)` хотя бы на 2/3 уровней; D_hf: `median(pos) > q75(neg)` хотя бы на 2/3 уровней |

Пороги консервативные для первого цикла. Могут быть пересмотрены после получения реальных распределений.

### Kill criterion

Если хотя бы одна метрика не проходит acceptance threshold:

1. **НЕ тюнить τ**
2. **НЕ вводить enforcement**
3. Диагностировать: какой тип negative cases плохо разделяется?
4. В зависимости от диагноза:
   - Если плохо разделяет D_parent → проверить выбор R или нормировку
   - Если плохо разделяет D_hf → проверить пару (R, Up) или переопределить positive/negative cases
   - Если оба → пересмотреть конструкцию метрик; возможно, нужен другой признак

### Что kill criterion запрещает

Подгонять τ под плохую separability. Это маскирует проблему вместо её решения.

---

## Шаг 5. Установка порогов (если acceptance пройден)

### 5.1 Data-driven τ_parent

```python
# По positive baseline:
τ_parent = quantile(D_parent[positive], q=0.95)

# Или с разбивкой по уровням:
τ_parent[L] = quantile(D_parent[positive, level=L], q=0.95)
```

### 5.2 Data-driven τ_hf (диагностический порог)

Аналогично τ_parent, но D_hf используется как сигнал, а не hard constraint.

### 5.3 Зависимость от энергии (опционально, второй проход)

Если в baseline обнаружена систематическая зависимость D_parent от ‖delta‖:
- На малых delta много false positives → рассмотреть adaptive τ_rel(E)
- На больших delta τ слишком мягкий → τ_rel(E) = убывающая функция энергии

Это второй проход, не первый.

---

## Шаг 6. Enforcement

После валидации и установки порогов:

```python
def check_scale_consistency(delta, level):
    R_delta = R(delta)  # R = gaussian_blur(sigma=3.0) + downsample
    D_parent = norm(R_delta) / (norm(delta) + eps)

    if D_parent > tau_parent[level]:
        return "REJECT"  # damp delta, reject split, increase local strictness
    return "OK"
```

D_parent как сигнал в ρ (контекстный, не самодостаточный):

```python
# Повышать приоритет probe если:
if D_parent > tau_warn AND residual > tau_residual AND gain > tau_gain_min:
    increase_probe_priority(node)

# НЕ поощрять split если:
if D_parent > tau_warn AND gain < tau_gain_min:
    # Проблема в механике refinement, не в структуре данных
    pass
```

---

## Связь с другими механизмами

| Механизм | Что защищает | Ортогональность |
|---|---|---|
| Halo | Локальные границы тайлов (геометрия). **NB:** применим только при boundary parallelism ≥ 3 и без context leakage (grid/graph: да, tree: нет) | Не пересекается со scale-consistency |
| SeamScore | Boundary artifacts внутри уровня | Не пересекается с D_parent |
| D_parent / D_hf | Межмасштабный семантический drift | Не пересекается с halo/seam |
| Probe | Защита от ложных fixed points | Дополняет scale-stable stop criterion |
| Budget governor | Глобальный расход | Ортогонален scale-consistency |

Scale-stable fixed point (локальная остановка):
```
gain_from_split < τ_gain
AND D_parent < τ_parent
AND устойчиво K шагов подряд
```

Probe остаётся обязательным даже при достижении fixed point.

---

## Что НЕ делать

- Не использовать learned Up на этапе верификации (вносит собственную семантику)
- Не менять пару (R, Up) в ходе одного цикла
- Не интерпретировать высокий D_parent как безусловный сигнал "split здесь"
- Не заменять D_parent + D_hf одним "великим объединённым индексом" без данных
- Не подгонять τ под плохую separability
