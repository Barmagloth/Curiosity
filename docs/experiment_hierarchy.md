# Curiosity — Иерархия экспериментов (v2.0)

Документ фиксирует актуальный статус, зависимости и порядок экспериментов.

Обновлено после Phase 0 (параллельная валидация: halo cross-space, SC-baseline, D_parent fix).

---

# Маппинг: папки → вопросы → план валидации

| Папка | Вопрос | План валидации | Статус |
|-------|--------|----------------|--------|
| `exp01_poc/` | Adaptive refinement работает? | — | ✅ Да (PoC) |
| `exp02_cifar_poc/` | (то же, CIFAR) | — | ✅ Да |
| `exp03_halo_diagnostic/` | Halo обязателен? | — | ✅ Да |
| `phase1_halo/` | Halo: r_min, blending, hardened | §A (A1+A2+A3) | ✅ Закрыт |
| `exp04_combined_interest/` | Комбинированная интересность нужна? | — | ✅ Да |
| `exp05_break_oracle/` | Oracle-free проверка | — | ✅ Да |
| `exp06_adaptive_switch/` | Авто-переключение ρ | — | ✅ Да |
| `exp07_gate/` | Двухстадийный гейт | — | ✅ Да |
| `exp08_schedule/` | Schedule + governor + probe | — | ✅ Закрыт |
| `phase2_probe_seam/` | Probe + SeamScore validation | §B (B1+B2) | ✅ Закрыт |
| `exp09a_layout_sandbox/` | Layout: grid vs compact (microbench) | §C (C3/Exp0.9a) | ✅ Частично |
| `halo_crossspace/` | Halo applicability across space types | Phase 0 | ✅ Закрыт (правило выведено) |
| `sc_baseline/` | Scale-consistency D_parent/D_hf verification | Phase 0 (SC-0..SC-4) | ✅ Закрыт (SC-5 open) |
| `p2a_sensitivity/` | Sensitivity sweep порогов гейта | P2a | 🔓 Код готов, не запущен |
| *(будущее)* | Layout GPU end-to-end | §C → P0 (0.9b0+) | 🔓 Открыт |

**Примечание:** §A/B/C — секции плана валидации, написанного между Exp0.3 и Phase 1.
В §B «B1/B2» = probe-сцены. В P1 ниже «B1/B2/B3» = компрессия дерева. Контексты разные.

---

# Закрыто (не требует экспериментов)

| # | Вопрос | Статус | Источник |
|---|--------|--------|----------|
| 1 | Adaptive refinement работает? | **Да** | PoC, Exp0.1–0.2 |
| — | Halo обязателен? | **Да**, w ∈ [2,4], cosine feather | Exp0.2–0.3, Phase 1 |
| — | Probe обязателен? | **Да**, uncert, 5–10% бюджета | Exp0.8, Phase 2 |
| — | SeamScore как production-метрика | **Да**, dual check работает в 4 пространствах | Phase 2 |
| — | Governor (EMA) для бюджета | **Да**, StdCost −50%, penalty −85% | Exp0.8 |
| 2 | Комбинированная интересность нужна? | **Да, при деградации сигнала.** Двухстадийный гейт | Exp0.4–0.7b |
| 3 | Phase schedule по глубине? | **Нет** при текущих условиях | Exp0.8v5 |
| — | Halo cross-space applicability | **Правило выведено** (grid/graph: да, tree: нет). boundary parallelism >= 3 AND no context leakage | Phase 0 |
| — | Morton layout | **Отложен (deferred)** (sort overhead, zero compute benefit) | 0.9a sandbox |
| — | Block-sparse layout | **Отложен (deferred)** (expansion ratio) | 0.9a sandbox |
| — | Детерминизм non-overlapping writes | **Чисто** (bitwise match) | 0.9a sandbox |

---

# Открыто — актуальная иерархия

## Уровень 0: инфраструктурные предусловия

Без ответов на эти вопросы результаты всего выше — декоративные.

### P0. Layout на GPU: grid vs compact

Единственный выживший кандидат. Зависимостей вверх нет — это фундамент.

```
P0. Layout (grid vs compact на GPU)
├── 0.9b0: buffer-scaling probe — O(k) vs O(M) буферы, synthetic kernel
│         kill/go для compact
├── 0.9b:  end-to-end pipeline (если compact жив)
│         grid vs compact, 4 пространства × 2 бюджета
│         метрики: VRAM, wall-clock, #kernels, SeamScore identity
├── 0.9c:  масштабирование (если compact жив)
│         sweep по M, clustered + random
└── 0.9h:  halo overlap детерминизм на GPU
          overlapping writes → порядок accumulation
```

**Выход P0:** фиксированный layout для всего дальнейшего. Либо grid (скорее всего), либо compact (если O(k) буферы спасают).

---

## Уровень 1: представление маршрутов

Зависит от P0 (layout определяет, как active_idx попадает в pipeline). Не зависит от "смысла" дерева — чистая инженерия хранения.

### P1. Компрессия и обслуживание структуры

```
P1. Компрессия дерева / маршрутов
├── B2: dirty-сигнатуры (12 бит: seam_risk + uncert + mass)
│       debounce (2 последовательных попадания)
│       сценарии: шум / структурное событие / drift
│       метрики: blast radius, latency-to-trigger, burstiness
│       ↑ это фундамент — без него B1 и B3 не знают когда запускаться
│
├── B1: segment compression (degree-2 + signature-stable + length cap)
│       зависит от B2 (критерий склейки = стабильность сигнатуры)
│       метрики: память vs node-per-node, стоимость локальных апдейтов
│
└── B3: anchors + periodic rebuild
        зависит от B1 + B2
        сценарий: частые локальные апдейты
        сравнение: (a) только локально (b) локально + periodic rebuild
        метрики: суммарная стоимость за N шагов, накопление "грязи"
```

**Выход P1:** формат хранения дерева (flat nodes vs segments), механизм dirty detection, стратегия rebuild.

---

## Уровень 2: надёжность комбинированного сигнала

Зависит от P0 (pipeline работает), не зависит от P1 (компрессия ортогональна).

### P2. Автонастройка и устойчивость ρ

Двухстадийный гейт подтверждён (Exp0.7b), но пороги (instability, FSR) — ручные.

```
P2. Автонастройка ρ-гейта
├── P2a: sensitivity analysis — sweep порогов instability/FSR
│        на существующих сценах (clean/noise/blur/spatvar/jpeg)
│        вопрос: насколько плоский "хребет" оптимальности?
│        если широкий → ручные пороги ок, автонастройка не нужна
│        если узкий → нужен adaptive threshold
│
└── P2b: adaptive threshold (только если P2a показал узкий хребет)
         online estimation instability/FSR percentiles
         метрики: PSNR stability across scenes, overhead
```

**Выход P2:** либо "ручные пороги ± 30% — без разницы" (и вопрос закрыт), либо конкретный механизм автонастройки.

---

## Уровень 3: семантика дерева

Зависит от P0 (layout) + P1 (формат хранения).

### P3. Даёт ли дерево смысловую метрику?

```
P3. Семантика дерева
├── P3a: LCA-расстояние как feature
│        на реальном дереве из pipeline: коррелирует ли LCA-distance
│        с "семантической близостью" (‖feature_i − feature_j‖)?
│        если нет → дерево = только журнал, не метрика
│
├── P3b: кусты (bushes) — кластеры путей
│        есть ли естественные кластеры среди leaf-путей?
│        метрики: silhouette, stability across runs
│
└── C-pre: проверка дискретности "профилей"
           trajectory features (EMA квантили, split signatures, stability)
           вопрос: есть ли кластерная структура?
           если да → C размораживается
           если нет → C мёртв
```

**Выход P3:** либо "дерево — чисто журнал, смысловую метрику не даёт" (ок, не страшно), либо конкретный способ извлечения семантики.

---

## Уровень SC: Scale-Consistency Invariant (v1.7)

Частично формализует мета-вопрос v1.5 «как не сломать фичи». Зависит от P0 (pipeline), не зависит от P1/P2/P3. Может идти параллельно.

### SC-baseline. Верификация метрик D_parent / D_hf

```
SC-baseline. Scale-Consistency Verification
├── SC-0: фиксация пары (R, Up), проверка идемпотентности R              ✅ COMPLETE
├── SC-1: подготовка positive (strong + empirical) и negative baseline    ✅ COMPLETE
├── SC-2: вычисление D_parent, D_hf по всем случаям                      ✅ COMPLETE
├── SC-3: анализ separability (AUC, effect size, quantile separation)     ✅ COMPLETE
│         глобально + по уровням + по типам структуры
├── SC-4: kill criterion — PASSED с обновлённой формулой D_parent
│         (R=gauss σ=3.0, lf_frac normalization, AUC=0.853, d=1.491)     ✅ COMPLETE
└── SC-5: установка data-driven τ_parent[L] — нужна data-driven настройка порогов
```

**Kill criterion:** Global ROC-AUC >= 0.75, Depth-conditioned AUC >= 0.65, Effect size >= medium (d >= 0.5). Если не проходит — менять метрики, **не** подгонять пороги.

**SC-4 результат:** PASSED. Обновлённая формула D_parent: `||R(delta)|| / (||delta|| + epsilon)`, R=gauss sigma=3.0. AUC=0.853, d=1.491. Cross-space: T1=1.000, T2=1.000, T3=1.000, T4=0.824 (все >= 0.75).

**SC-5 статус:** tau_parent нуждается в data-driven threshold setting.

**Выход SC-baseline:** валидированные пороги tau_parent[L] или решение о пересмотре конструкции метрик.

Полный протокол: `docs/scale_consistency_verification_protocol_v1.0.md`.

### SC-σ sweep. Оптимизация параметра σ оператора R

```
SC-σ. Fine-grained sweep параметра σ
├── σ sweep: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
│       на каждом σ: полный SC-3 (AUC, Cohen's d, per-negative-type)
├── tile_size sweep: σ × tile_size ∈ {8, 16, 32, 64}
│       вопрос: σ_opt зависит от tile_size? Есть ли σ/tile_size ≈ const?
├── cross-space: σ sweep × 4 типа пространств
│       вопрос: σ_opt одинаков для всех пространств или space-dependent?
└── вывод: формула/правило для выбора σ, или фиксированный σ_opt
```

**Known limitation текущего σ=3.0:** выбрано как наименьшее целое в грубом sweep [0.5, 1.0, 2.0, 3.0]. Fine-grained поиск не проводился. Возможно σ=2.5 достаточно, или σ=4.0 лучше. Оптимум может зависеть от tile_size и типа пространства.

**Зависимости:** нет (можно запускать параллельно с чем угодно, переиспользует код sc_baseline).
**Приоритет:** низкий (σ=3.0 проходит kill criteria; оптимизация — не блокер).

---

### SC-enforce. Enforcement (после SC-baseline)

```
SC-enforce. Scale-Consistency Enforcement
├── damp delta / reject split / increase local strictness при D_parent > τ_parent
└── D_parent как контекстный сигнал в ρ (не самодостаточный)
```

---

## Уровень 4: глобальная согласованность ("не сломать фичи")

Зависит от **всего выше** + SC-baseline. Мета-вопрос из Concept v1.5, частично формализован через Scale-Consistency Invariant (Concept v1.7, раздел 8).

### P4. Согласованность представления при неоднородной глубине

```
P4. "Не сломать фичи"
├── P4a: downstream consumer test
│        подать adaptive-refined представление в простой downstream
│        (классификатор / автоэнкодер)
│        сравнить с dense-refined и coarse-only
│        вопрос: ломается ли downstream при неоднородной глубине?
│        (с enforcement scale-consistency vs. без)
│
├── P4b: matryoshka invariant
│        проверить что представление на любом уровне "матрёшки"
│        валидно как вход для потребителя
│        (не только визуально гладко, но функционально корректно)
│
└── P4c: механизм гарантии (если P4a/b показали проблему)
         варианты: padding/projection слой, consistency loss,
         depth-aware normalization, усиление τ_parent
```

**Выход P4:** либо "неоднородная глубина не ломает downstream" (и вопрос закрыт), либо конкретный механизм защиты.

---

# Заморожено

## C. DAG + профили

**Входной контракт (все три одновременно):**

1. Минимум две несводимые цели (не сводятся в скаляр без потери семантики)
2. Конкретный downstream consumer, который на этих целях реально стоит
3. Наблюдаемый конфликт: разные оптимальные решения при разных целях на одних данных

Предэксперимент C-pre (в P3) может дать сигнал к размораживанию, но сам по себе не достаточен.

Если контракт не выполнен — заморозка бессрочная.

---

# Граф зависимостей

```
P0 (layout GPU)
 ├──→ P1 (компрессия дерева)  ──→ P3 (семантика дерева)
 │                                       │
 ├──→ P2 (автонастройка ρ)               ├──→ C-pre
 │                                        │
 └──→ SC-baseline (✅ SC-0..SC-4) ──→ SC-5 ──→ SC-enforce ──→ P4 ("не сломать фичи")
                                           зависит от P0 + P1 + P2 + P3 + SC
```

**Критический путь:** P0 → P1 → P3 → P4.

**Параллельные ветки:** P2, SC-baseline — обе идут параллельно P1, все нужны до P4.

---

# Рабочий порядок

1. **P0: 0.9b0** — buffer-scaling probe, kill/go для compact
2. **P0: 0.9b/0.9c/0.9h** — если compact жив; иначе фиксируем grid
3. **P1-B2** — dirty-сигнатуры (параллельно с P0 GPU-частью на CPU)
4. **P2a** — sensitivity sweep порогов гейта, **5 сцен × 4 пространства** (код готов, параллельно с P1)
5. **SC-5** — установка data-driven τ_parent[L] (SC-0..SC-4 ✅ завершены; параллельно с P1)
6. **P1-B1** — segment compression (после B2)
7. **P1-B3** — anchors + rebuild (после B1+B2)
8. **SC-enforce** — enforcement scale-consistency (после SC-5)
9. **P3a/P3b** — семантика дерева (после P1)
10. **C-pre** — кластерность профилей (после P3, дёшево)
11. **P4** — "не сломать фичи" (после всего + SC)

---

# Конвенция нумерации (v3+)

Исторически нумерация росла стихийно: хронологические номера (0.1–0.9a),
буквенные секции плана валидации (§A/B/C), уровни roadmap (P0–P4),
суб-эксперименты внутри уровней (B1–B3 в P1). Результат — путаница.

**Правила для новых экспериментов:**

1. **Единая сквозная нумерация.** Следующий эксперимент = `exp10`.
   Нумерация целочисленная, без точек (точка путалась с подверсиями).
   Номер = порядок создания. Никогда не переиспользуется.

2. **Суб-эксперименты — строчная буква.** `exp10a`, `exp10b`, `exp10c`.
   Одна серия = один числовой корень.

3. **Папка = `exp{N}{суффикс}_{краткое_имя}/`.** Например:
   `exp10_buffer_scaling/`, `exp10a_synthetic_kernel/`, `exp11_dirty_signatures/`.

4. **Маппинг на roadmap — только в этом документе**, не в именах папок.
   Папка не содержит «P0» или «B2» в названии.

5. **Каждая папка содержит README.md** (короткий, 5–15 строк):
   - Вопрос/гипотеза (одно предложение)
   - Kill criteria
   - Ссылка на уровень roadmap (P0/P1/P2/...)
   - Статус (open / closed / killed)

6. **Старые имена не переименовываются.** `phase1_halo/`, `phase2_probe_seam/`,
   `exp09a_layout_sandbox/` — историческое наследие, связь с новой нумерацией
   зафиксирована в таблице маппинга выше.

**Маппинг рабочего порядка → номера экспериментов:**

| Шаг | Roadmap | Описание | Будущий exp# |
|-----|---------|----------|-------------|
| 1 | P0 | buffer-scaling probe (kill/go compact) | exp10 |
| 2 | P0 | end-to-end pipeline grid vs compact | exp10a/b/c |
| 3 | P1-B2 | dirty-сигнатуры | exp11 |
| 4 | P2a | sensitivity sweep порогов гейта (5 сцен × 4 пространства) | exp12 |
| 5 | SC-5 | установка data-driven τ_parent[L] (SC-0..SC-4 ✅) | exp12a |
| 6 | P1-B1 | segment compression | exp13 |
| 7 | P1-B3 | anchors + rebuild | exp14 |
| 8 | SC-enforce | enforcement scale-consistency | exp14a |
| — | SC-σ | fine-grained σ sweep × tile_size × 4 пространства (низкий приоритет) | exp14b |
| 9 | P3a/b | семантика дерева | exp15 |
| 10 | C-pre | кластерность профилей | exp16 |
| 11 | P4 | «не сломать фичи» | exp17 |

Номера предварительные. Если между шагами возникнет незапланированный
эксперимент — он получает следующий свободный номер.

---

# Instrument Readiness Gate

Все эксперименты P0–P4 + SC относятся к **Track A** (построение инструмента). Переход к **Track B** (исследование структуры дерева) — только после прохождения Instrument Readiness Gate:

1. **Invariant pass** — все обязательные инварианты выполняются
2. **Overhead profile** — overhead не съедает выигрыш
3. **Stability pass** — система устойчива между прогонами
4. **One validated benchmark** — adaptive > random > coarse с подтверждёнными числами
5. **Attribution diagnostics** — вклад каждого модуля измерен (ablation)

Подробно: `docs/target_problem_definition_v1.1.md`.

После успешной Track B открывается **Track C** (обобщение на нон-пространственные домены: графы, латентные пространства, активации). Долгосрочная амбиция, не текущая цель.

---

# Принципы

* Сначала судья-цифры, потом амбиции.
* Ни один "следующий этап" не фиксируется заранее.
* Kill criteria — двусторонние (speed + memory).
* Forensic-grade протокол: controls, Holm-Bonferroni, cost-fair comparisons.
