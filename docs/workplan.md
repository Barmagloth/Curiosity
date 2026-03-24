# План работ (Curiosity)

> **Статус:** Модули A-H описаны. По состоянию на 23 марта 2026:
> - Модуль C (интересность ρ): валидирован (Exp0.4-0.7, двухстадийный гейт)
> - Модуль D (дерево + split/merge): валидирован (Exp0.1-0.3, Exp0.8 governor)
> - Модуль E (дельты + границы): валидирован (halo w∈[2,4], SeamScore, Phase 1/2)
> - Модуль F (бенчмарк): проведён (серия exp10, 158K+ trials, layout policy зафиксирована)
> - Модуль G (Scale-Consistency): валидирован (SC-baseline AUC 0.82-1.0, exp12a τ_parent PASS)
> - **Модуль H (трёхслойная ρ): валидирован** (exp17, 1080 конфигов, reusability 12/12 PASS, cascade quotas, streaming pipeline)
> - Модули A, B (каноникализация, кэш): НЕ реализованы (не на критическом пути)
> - Phase 1 завершена (20.03.2026). Phase 2 завершена (21.03.2026). Phase 3 завершена (22.03.2026). Phase 3.5 завершена (23.03.2026).
> - Следующий шаг: Phase 4 (P4a downstream, P4b matryoshka) + C-оптимизация scoring (roadmap).

## Базовая логика
1. Выживают только модули, которые дают выигрыш **сами по себе**: кэш, детектор, планировщик пересчёта, профилирование.
2. Старт — не с «умного дерева», а с **идентичности данных**: каноникализация → хэш → кэш.
3. Дерево без стабильности и метрик — декоративный куст. Сначала правила изменения и измерения, потом структура.

---

## A. Каноникализация + контентный хэш тайлов (первый живучий кирпич)
**Цель:** любой участок данных (tile/patch/block) получает стабильный ID и хэш «смысла», а не «как легло в память».

**Деливераблы:**
- `RegionPath`: путь в дереве (root → quadrants), стабильная сериализация в байты.
- `TileSpec`: shape, dtype, нормализация contiguous/stride, padding-policy.
- `Hash(tile)`: одинаковые данные → одинаковый хэш независимо от аллокаций.
- Таблица: `hash -> cached_result` + счётчики hit/miss.

**Тесты:**
- Один и тот же тайл в разных буферах → одинаковый хэш.
- Микроизменение данных → хэш меняется (если не включён отдельный режим «мягкого» хэша).

**Польза отдельно:** content-addressable cache для любого пайплайна.

---

## B. Планировщик пересчёта «по изменению» (инкрементальность)
**Цель:** не делать работу, если вход не изменился.

**Деливераблы:**
- API: `compute(tile) -> result`, обёртка `cached_compute(tile, key=hash)`.
- Очередь задач: пересчитываем только изменившиеся тайлы.
- Журнал: какие тайлы пересчитаны и почему.

**Метрики:**
- Доля пересчитанных тайлов при локальном изменении (ожидаемо мала).
- Latency/throughput с кэшем и без.

**Польза отдельно:** инкрементальный пересчёт как самостоятельная оптимизация.

---

## C. «Интересность» как измеряемая функция (пока без деревьев)
**Цель:** определить «где считать» через измерения, а не через веру.

**Деливераблы:**
- Набор скорингов на тайле:
  - энергия градиента / высоких частот,
  - ошибка реконструкции (если есть baseline/teacher),
  - дисперсия активаций/градиентов.
- Нормализация скорингов к сопоставимой шкале.
- Пороги не «из головы», а из квантилей распределения.

**Польза отдельно:** ROI-маска внимания для компрессии, логирования, адаптивного апсемплинга и т.п.

---

## D. Дерево областей (quadtree/octree) + правила split/merge
**Цель:** формализовать разбиение пространства, чтобы планировщик работал адресно.

**Деливераблы:**
- Узел: `path, level, bbox, hash, score, state`.
- Операции:
  - `split(node) -> 4 children`,
  - `merge(children) -> parent`.
- Стабильность:
  - гистерезис (разные пороги split/merge),
  - минимальный «возраст узла» до схлопывания,
  - лимиты на глубину и/или скорость изменений.

**Польза отдельно:** адаптивная разметка сложности сцены/данных.

---

## E. Пересчёт только дельт и только на границах
**Цель:** новые листья считают уточнение, старые не трогаем, границы согласовываем.

**Деливераблы:**
- Явное определение «delta» (например, residual относительно coarse уровня).
- Политика пересчёта:
  - `new_leaves`,
  - `changed_hash_leaves`,
  - `boundary_neighbors`.
- Стыковка уровней:
  - overlap/padding,
  - blending на границах (без швов).

**Польза отдельно:** boundary-only refinement применим как независимая оптимизация.

---

## F. Бенчмарк + стоимость управления (анти-самообман)
**Цель:** измерить, не съедает ли выигрыш оверхед дерева/очередей/хэшей.

**Деливераблы:**
- Профилирование по стадиям: hashing / scheduling / compute / merge.
- Баланс-карта: время и память по компонентам.
- Сценарии:
  - гладкая сцена (должно быть дёшево),
  - один резкий объект (дробление локально),
  - шум (не дробить бесконечно).

---

## G. Scale-Consistency: не сломать семантику родительского масштаба (v1.6; обновлён v1.7)
**Цель:** гарантировать, что delta не переопределяет coarse-уровень «контрабандой» — refinement добавляет детали, но не проталкивает новый LF-смысл наверх.

**Деливераблы:**
- Реализация пары операторов (R, Up): `gaussian blur + decimation` / `bilinear upsampling`.
- Вычисление метрик D_parent и D_hf по каждому узлу дерева.
- Baseline-эксперимент: собрать распределения D_parent/D_hf на positive (корректный refinement) и negative (искусственный LF drift) случаях. Оценить separability (AUC, effect size, quantile separation).
- Data-driven пороги τ_parent[L] по результатам baseline.
- Enforcement: damp delta / reject split / increase local strictness при D_parent > τ_parent.
- Интеграция D_parent как контекстного сигнала в ρ (не самодостаточного).

**Kill criterion:** если separability между positive и negative недостаточна — пересматриваются метрики или пара (R, Up), а не пороги.

**Протокол:** `scale_consistency_verification_protocol_v1.0.md`.

**Польза отдельно:** диагностика межмасштабной семантической когерентности для любого hierarchical representation.

---

## Мини-роадмап (каждый шаг — отдельный трофей)
1. Hash+Cache для тайлов (каноникализация, стабильный ключ, hit/miss, журнал).
2. Инкрементальный планировщик (пересчитываем только изменившееся).
3. Скоринг «интересности» (метрики + нормализация + квантильные пороги).
4. Дерево + гистерезис (split/merge без дрожи).
5. Политика пересчёта (new/changed/boundary) + измерение выигрыша.
6. Delta + boundary stitching (overlap/blend, чтобы не было швов).
7. **Детерминизм** (canonical traversal order, deterministic probe, governor isolation → побитовая воспроизводимость при фиксированном seed + статистическая устойчивость по seeds).
8. Scale-Consistency baseline (R/Up, D_parent/D_hf, separability, τ_parent).
9. Scale-Consistency enforcement (damp/reject/strictness + интеграция в ρ).
10. Оптимизации (батчинг, ядра, GPU-специфика) — только после стабилизации.

---

## Что останется полезным, даже если «всё целиком» не выгорит
- Content-addressable cache.
- Инкрементальный пересчёт.
- ROI-маска «интересности».
- Стабильная адаптивная разметка сложности.
- Диагностика межмасштабной когерентности (D_parent / D_hf).

---

## Phase 4: MultiStageDedup testing (планируется)

**Цель:** активировать MultiStageDedup (3-уровневая дедупликация из Enox-инфраструктуры) с epsilon > 0 в multi-pass/iterative refinement режиме. В текущем single-pass (epsilon=0.0) ни один уровень не срабатывает — это заготовка для Phase 4.

**Зависимости:** P4a (downstream consumer test) — multi-pass контекст необходим для того, чтобы деdup имел смысл.

**Kill criteria:** dedup сокращает бюджетный расход на повторно обрабатываемые юниты без потери PSNR > 0.5 dB.

---

## H. Трёхслойная декомпозиция rho (exp17, 22-23 марта 2026)

**Статус:** экспериментально валидирован. Архитектура работает, reusability PASS на всех пространствах и масштабах (100 / 1K / 10K).

### Суть

Монолитная rho заменяется каскадом из трёх слоёв:

```
Layer 0: ТОПОЛОГИЯ       — "как устроено пространство" (data-independent)
Layer 1: ДАННЫЕ ДА/НЕТ   — "где в пространстве что-то лежит" (data-dependent, query-independent)
Layer 2: ЗАПРОС           — "из того что лежит — где нужное мне" (task-specific)
```

Каждый слой сужает рабочее множество для следующего. Переиспользуемость растёт снизу вверх: топология неизменна, карта данных обновляется редко, запрос меняется постоянно.

### Ключевые архитектурные решения

1. **Каскадные квоты (Variant C):** L1 не режет по фиксированному порогу — каждый L0-кластер гарантирует минимальное число выживших, пропорционально размеру кластера. Порог выживаемости привязан к budget_fraction. Топология диктует квоты. Ни один регион не вымирает.

2. **Streaming pipeline:** вместо L0(all) → L1(all) → L2(all) — обработка покластерно: каждый L0-кластер проходит L0 → L1 → L2 до перехода к следующему. Кластеры обрабатываются в порядке приоритета L0 score. Преимущества:
   - Первые результаты появляются после 1 кластера, а не после полной карты.
   - L1 pruning реально сокращает количество refinements (budget per-cluster).
   - Можно остановить досрочно — частичные результаты уже полезны.

### Экспериментальные результаты (exp17, 1080 конфигов)

- **Reusability:** 12/12 PASS (min ratio = 0.838, threshold = 0.80). Frozen tree переиспользуем для разных запросов.
- **PSNR:** на 2-4 dB ниже single_pass/kdtree на grid (цена L1 pruning), паритет на graph/tree.
- **Время (single query):** streaming быстрее batch на 10-20%, но kdtree (scipy C) быстрее обоих.
- **Amortized:** three_layer выигрывает у single_pass начиная с 2 запросов на tree_hierarchy.

### Roadmap: C/Cython оптимизация (Phase 5+)

**Обоснование:** текущий bottleneck — Python overhead в scoring фазах. Refinement (numpy) уже near-C. Переписывание scoring на C даст мультипликативный эффект именно для streaming, потому что streaming scoring'ит МЕНЬШЕ юнитов.

**Прогнозируемые speedups (C vs Python):**

| Компонент | Python (сейчас) | C (прогноз) | Speedup |
|-----------|----------------|-------------|---------|
| L0 topo extraction (graph) | 70ms | ~2-5ms | 15-35x |
| L1 presence scoring | 5ms | ~0.5ms | 10x |
| L2 query scoring | 8ms | ~0.8ms | 10x |
| Refinement (numpy) | 19ms | ~19ms | 1x (уже быстрый) |
| **TOTAL streaming (1K grid)** | **33ms** | **~22ms** | **1.5x** |
| **TOTAL streaming (graph)** | **68ms** | **~22ms** | **3x** |

**Ожидаемый итог:** streaming на C обойдёт kdtree на graph/tree пространствах и сравняется на grid. При этом сохранит побочные данные (topo features, zones, cluster structure, decision journal), которые kdtree не даёт.

**Приоритет реализации:**
1. L0 topo на C/Cython (максимальный ROI — 70ms → 5ms)
2. L1 + L2 scoring на C (единый vectorized pass)
3. Батчинг refinement (группировка смежных тайлов для cache locality)

---

## Будущие исследования (пост-Phase 4)

### RG-flow verification

RG-flow верификация (пост-Phase 4): basin membership требует multi-pass для формирования бассейнов. Exp18 показал r=0.019 в single-pass (FAIL). Связано с: Exp0.10 (R,Up) sensitivity (concept section 8.10).

### Governor EMA restoration + sweep

Восстановить EMA feedback из exp0.8 как глобальный термостат strictness (потерян при сборке Phase 2 pipeline). Двухслойная архитектура: hardware parameter задаёт диапазон, EMA feedback управляет внутри диапазона. Применим в batch и frozen reuse; в streaming НЕ применим (cross-cluster bleed).

### Convergence detector (отсутствует)

**Обнаружено:** визуализация рантайма (viz/index.html) показала, что после уточнения всех полезных тайлов система продолжает крутить пустые тики — бюджет не исчерпан, но кандидатов нет или все ниже порога. Governor регулирует *интенсивность* уточнения (strictness), но не принимает решение *остановиться*.

**Проблема:** нет формального convergence detector. Текущие stop-условия:
- Budget exhaustion (hard stop)
- Waste ≥ R_max (force-stop текущего тика, но не сессии)
- Нет: "accepted == 0 последние K тиков → stop session"

**Требуется:**
```
Convergence rule:
  IF accepted == 0 for last K ticks (K ≥ 2):
    STOP session, return remaining budget as unspent
  IF accepted < probe_count for last K ticks:
    STOP session (only probe discovers anything = diminishing returns)
```

**Зависимость:** фиксировать при сборке Phase 4 pipeline. Не требует новых экспериментов — чисто инженерное решение на уровне harness loop.

### Gate health thresholds — data-driven калибровка (открытая проблема)

**Обнаружено:** при реализации двухступенчатого gate в визуализации (viz/index.html) выяснилось, что пороги instability/FSR для переключения Stage 1 ↔ Stage 2 не могут быть фиксированными. Захардкоженные значения из P2a sweep (0.3/0.2) не работают на данных с другой статистикой — instability metric (CV residual) масштабируется по-разному в зависимости от данных, tile size, стадии refinement и уровня шума.

**Текущее состояние в коде:** P2a sweep дал точечные значения для тестовых полей. Не generalization.

**Пороги зависят от:**
- Шум данных — больше шума → порог relaxed (иначе Stage 1 никогда не включится)
- Гетерогенность поля — однородное поле → CV naturally низкий → порог должен быть ниже
- Tile size — крупные тайлы → больше averaging → ниже CV
- Стадия refinement — на первых тиках residual нестабилен, к концу стабилизируется

**Решение — композитный подход (два механизма одновременно):**

**(A) Pilot calibration.** Первые K тиков (K=2-3) — всегда Stage 2 (сбор статистики). После pilot: threshold = percentile(observed) × factor. Даёт абсолютный порог, откалиброванный под конкретные данные. Работает в viz: при seed=42 pilot даёт instabThresh=1.295, после чего gate осциллирует на границе Stage 1/2 — реалистичная динамика.

**(B) Relative threshold.** Переключаться в Stage 1 когда instability < EMA(instability за последние K тиков) × factor. Адаптируется к тренду: если instability стабильно падает (refinement прогрессирует) — порог подтягивается вниз, не застревает на pilot-калиброванном значении.

**Композит A+B:** порог = min(pilot_threshold, ema_relative_threshold). Pilot задаёт верхнюю границу (worst-case из начальных данных). EMA подтягивает вниз по мере стабилизации. Это предотвращает и слишком раннее переключение (до калибровки), и застревание в Stage 2 когда данные давно стабильны.

**Зависимость:** реализовать при сборке Phase 4 pipeline. Требует sweep для валидации factor в A и B. Визуализация (viz/) может быть использована как testbed.

### FSR metric — refined tiles inflate sign-flip rate (баг)

**Обнаружено:** viz стенд. FSR (fraction of sign-flips) считает все тайлы, включая только что уточнённые. Уточнённый тайл → residual≈0 → гарантированный sign-flip. При 40 тайлах/тик из 1024 это ~4% baseline FSR даже на идеальных данных. Если fsrThresh < 4% → gate застревает в Stage 2 навсегда.

**Фикс:** FSR должен считаться **только по non-refined тайлам**. Реализовано в viz, нужно проверить реальный pipeline (exp06, exp07).

**Зависимость:** проверить при сборке Phase 4.

### Gate oscillation — отсутствие hysteresis

**Обнаружено:** viz стенд. Когда instability ≈ threshold, gate прыгает Stage 1→2→1→2 каждый тик. Каждый тик использует разную ρ-функцию (residual-only vs combo), что нарушает DET-2 (metric stability across seeds): один seed может попасть на Stage 1, другой на Stage 2 в том же тике.

**Требуется:** hysteresis band. Переключение в Stage 1 при instab < threshold × 0.8, обратно в Stage 2 при instab > threshold × 1.2. Концепт упоминает "EMA-smoothing с hysteresis" (exp06-07), но неясно реализовано ли это в текущем коде.

**Зависимость:** Phase 4 pipeline. Проверить exp07b_twostage.py.

### EMA-веса не откалиброваны на первых тиках

**Обнаружено:** viz стенд. На тике 1 prevGains=null → EMA не может адаптировать веса → первые тики Stage 2 работают с произвольными начальными весами (0.4/0.35/0.25). Решения на их основе необратимы — уточнённые тайлы не отменишь.

**Проблема:** pilot calibration решает проблему **порогов**, но не проблему **весов**. Первые K тиков Stage 2 работают вслепую.

**Возможные решения:**
- Первый тик Stage 2 использовать equal weights, но пометить решения как low-confidence
- Использовать HF+Variance only на первом тике (они не зависят от предыдущих gains)
- Задержать commitment: pilot тики оценивают но не уточняют, только собирают статистику для калибровки весов

**Зависимость:** Phase 4. Связано с pilot calibration threshold — можно объединить в единую pilot phase.

### Gain threshold — несоизмеримость gain и cost

**Обнаружено:** viz стенд. При исчерпании хороших кандидатов (80%+ тайлов уточнено) относительный порог (quantile от текущих кандидатов) пропускает тайлы с мизерным gain, потому что порог relative. Абсолютный порог нужен, но gain (MSE) и cost (compute units) — несоизмеримые величины, прямое сравнение gain > cost не имеет смысла.

**Проблема шире:** в текущей архитектуре нет единой "валюты" для gain и cost. Governor оперирует compute units, ρ оперирует нормализованными scores, gain — это MSE. Три разных пространства.

**Возможные решения:**
- ROI metric: gain_MSE / cost_compute — безразмерный, сравнимый. Порог = min ROI для accept.
- Reference gain: порог = fraction × median_gain_tick1 (реализовано в viz как refGain × 0.15 × strictness). Хрупко, но работает.
- Decision-theoretic: expected_improvement / expected_cost > threshold, где threshold = f(remaining_budget, remaining_candidates).

**Зависимость:** Phase 4. Фундаментальный архитектурный вопрос.

### Probe пересекается с rejected tiles

**Обнаружено:** viz стенд. Probe таргетирует lowest-ρ невыбранные тайлы. Но governor может оценить и осознанно отклонить тайл (gain < threshold). Probe может нацелиться на тот же тайл — бессмысленная трата probe budget на тайл, который governor только что отверг.

**Фикс:** probe должен таргетировать **неоценённые** тайлы (не попавшие в evalCount), а не просто невыбранные. Probe = исследование неизвестного, не переоценка отвергнутого.

**Зависимость:** Phase 4 pipeline. Простой фикс на уровне probe selection logic.

### Нет обратной связи от качества refinement

**Обнаружено:** viz стенд. Тайл accepted → refined → помечен навсегда. Никто не проверяет, дало ли уточнение реальное улучшение. Если оператор (или в AdaHiMem: compression/decompression) плохо реконструирует — тайл считается "уточнённым" навсегда, governor не получает сигнал о плохом качестве.

**Проблема:** strictness per-tile учитывает только accept/reject решение, не post-refinement quality. Тайл с высоким gain но плохим refinement — blind spot.

**Возможные решения:**
- Verify-after-refine: после уточнения проверить residual на этом тайле. Если residual не уменьшился значимо → пометить как suspicious, увеличить strictness для соседей.
- Quality-weighted strictness: strictness decay пропорционален реальному improvement, а не просто факту accept.
- Rollback: если post-refinement quality < threshold → откатить, вернуть тайл в candidates pool, увеличить strictness.

**Зависимость:** Phase 4+. Требует дополнительного residual pass после refinement. Увеличивает compute cost на ~10-20% (один дополнительный residual на уточнённые тайлы).

### Noise-fitting — система оптимизирует к шуму, не к сигналу (фундаментальная проблема)

**Обнаружено:** viz стенд с шумом на исходном поле (σ > 0).

**Суть проблемы:** Curiosity минимизирует residual = |current - observed_data|. Если observed_data зашумлены, система будет тратить бюджет на точное воспроизведение шума. После "уточнения" тайла residual → 0, но quality не улучшилось — шум скопирован в результат. Система считает тайл "уточнённым" (residual=0, gain=0, больше не кандидат), хотя реальный сигнал восстановлен с ошибкой σ.

**Где это проявляется в текущей архитектуре:**

1. **ρ(x) слепа к шуму.** Все три сигнала (residual, HF, variance) вычисляются из наблюдаемых данных. На зашумлённых данных HF-энергия повышена (шум = high-frequency), variance повышена, residual смещён. ρ(x) считает зашумлённые области "интересными" и направляет бюджет на воспроизведение шума.

2. **Gain = MSE к наблюдаемым данным.** gain_marginal[tile] = MSE(current, observed). При шуме gain > 0 даже для плоских областей (шум создаёт ненулевую MSE). Governor принимает тайлы с gain > threshold, где gain — это "расстояние до шума", а не "расстояние до сигнала".

3. **Refinement = копирование шума.** Уточнение тайла заменяет coarse approximation на observed data. Если observed = signal + noise, то refined = signal + noise. Coarse approximation (downsampled) фактически была ближе к signal (downsampling = low-pass filter → noise reduction). Парадокс: **уточнение ухудшает quality на зашумлённых данных**, хотя residual падает.

4. **Scale-consistency invariant не защищает.** D_parent проверяет что delta не содержит low-frequency content. Шум — high-frequency. Delta = noise проходит SC-check без проблем.

5. **Two-stage gate адаптируется, но не спасает.** При шуме gate остаётся в Stage 2 (instability высокий), EMA-веса перевешивают с residual на HF+variance. Это правильная реакция на нестабильный residual, но не решает проблему: HF и variance тоже contaminated шумом.

**Почему это не поймали раньше:** все эксперименты (exp00a-exp18) работали с чистыми синтетическими данными или с данными где шум пренебрежимо мал. P2a sweep тестировал шум как добавку к ρ-сигналу (noise on signal scores), но не шум на исходных данных.

**Возможные решения:**

**(A) Noise floor rejection.** Не уточнять тайл если gain < estimated_noise_variance. Требует оценку дисперсии шума (можно из flat regions или из разницы соседних пикселей). Простой, но хрупкий — noise_floor estimation на реальных данных нетривиальна.

**(B) Regularized refinement.** Вместо refined = observed, использовать refined = denoise(observed). Например: local average, bilateral filter, или wavelet thresholding. Добавляет ещё один оператор в pipeline, но гарантирует что refinement не хуже coarse.

**(C) Oracle-free quality metric.** Вместо MSE к observed data использовать cross-validation: разбить тайл на две половины, уточнить одну, проверить предсказание на другой. Если prediction error не уменьшился → refinement бесполезно (шум не предсказуем). Дорого (2× compute), но honest.

**(D) Coarse-as-prior.** Сравнивать refined не с observed, а с coarse: если |refined - coarse| > expected_signal_variation → refined содержит шум. Тогда dampening: refined = α×observed + (1-α)×coarse, где α адаптивен. По сути — Bayesian: coarse = prior, observed = likelihood.

**Зависимость:** фундаментальная проблема. Не Phase 4 — скорее Phase 5 (robustness). Но нужно как минимум добавить noise-awareness в quality metrics, чтобы Phase 4 эксперименты не давали ложноположительных результатов на шумных данных.

### Streaming budget control (B+C)

Плавное управление бюджетом для streaming mode (сейчас только бинарные go/stop):
- **(B) L0-informed allocation (формула институционального неравенства).** Вес кластера в бюджете:

  $$W_{cluster} = N_{units} \times (1 - ECR)^{\gamma}, \quad \gamma \geq 2$$

  Вывод из термодинамики StrictnessTracker: $$E[\Delta S] = (1 - ECR) \times 0.9 + ECR \times 1.5$$. GREEN (ECR=0.05): E[ΔS]=0.93 → процветает. YELLOW (ECR=0.15): E[ΔS]=0.975 → равновесие. RED (ECR=0.33): E[ΔS]=1.098 → умирает от WasteBudget. Квадратичная (γ=2) соответствует реальной пропускной способности: GREEN ~90%, YELLOW ~72%, RED ~42% номинала. γ валидируется в sweep: γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} (от линейного baseline до агрессивного).

- **(C) Adaptive redistribution:** неиспользованный бюджет кластера перетекает к следующим (forward carry). RED получает строгий минимум, но при аномально чистом прогоне получает остатки от GREEN.

### Governor + B+C sweep test

Sweep: 3 режима (batch/reuse/streaming) × 3 hardware profile (low/mid/high) × 6 γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} × 4 spaces × 20 seeds. Метрики: PSNR, time, reject rate, compliance, budget utilization. Kill: batch/reuse — EMA улучшает compliance; streaming — B+C(γ*) ≥ equal-allocation baseline.

### Content-Addressable Frozen Tree Cache (после валидации RegionURI Hash)

**Зависимость:** RegionURI Hash (из Enox ADOPT-списка) должен быть валидирован как стабильный ключ кеширования.

**Идея:** Frozen Tree — уже переносимый артефакт (l0_scores, l1_scores, active_units, zone, memory footprint). Добавить content-addressable хеширование по аналогии с Bazel/CAS: одинаковые входные данные (region + параметры) дают детерминированный hash, frozen tree сохраняется по этому ключу и переиспользуется без пересчёта.

**Шаги:**
1. Валидация RegionURI Hash как ключа (детерминизм, коллизии, стабильность при разных seeds).
2. Реализация portable cache store (файловый CAS: hash -> frozen tree blob).
3. Cache hit/miss метрики + измерение реального выигрыша по времени.
4. Shared cache между сессиями/машинами (если выигрыш > 2x на cache hit).

**Kill:** cache hit rate < 20% на типичных workloads ИЛИ overhead хеширования > выигрыша от кеша.

### Code Domain Application (исследование)

**Принцип:** начинать с инвариантов Curiosity (identity, distance, rho, budget, refinement), а не с выбора структуры данных. Носитель вторичен — сигнал, бюджет, локальность и уточнение первичны.

**Правильная постановка:** для кодовой базы нужно выбрать семейство пространств X, на которых можно стабильно определить identity, distance, rho, budget и refinement.

#### Три варианта пространства

**A. Код как дерево** — refinement = углубление по иерархии (repo > package > file > class > function > block). split = спуск на уровень, distance = LCA/общий префикс, hash = структурный хеш поддерева, probe = выборочные углубления в тихих ветках. Самый естественный перенос (дерево как журнал уточнения уже зафиксировано в концепции).

**B. Код как граф** — refinement зависит от связности (call graph, dependency graph, inheritance, data-flow, test-impact). split = локализация внутри подграфа, distance = hops/weighted path/co-change affinity, boundary = интерфейсы между подграфами. Другой носитель топологии, не "правильнее" дерева.

**C. Код как гибридное пространство (рекомендуемый):**
- tree gives identity (адресация и кеш)
- graph gives interaction (сигналы риска и влияния)
- budget gives decision (стоимость анализа/тестов/рефакторинга)
- change history как дополнительная геометрия поверх всего

#### Parser stack (не один parser)

1. AST / иерархия символов — для дерева регионов
2. dependency / call relations — для графовой топологии
3. git / change history — для эмпирической близости
4. test / runtime coverage — для поведенческого слоя

#### Метрики (пространственно-зависимые)

Дерево: subtree size, depth, branching irregularity, structural churn, hash invalidation rate.
Граф: degree/fan-in/fan-out, SCC/cycles, betweenness/centrality, cut edges/bridge nodes, coupling density.
Гибрид: structural anomaly, graph influence, churn/bug history, uncertainty, expected payoff vs cost.

#### Семейство расстояний (не одно)

- tree distance: LCA, глубина общего предка, путь в иерархии
- graph distance: shortest path, weighted path, centrality-aware proximity
- change distance: как часто менялись вместе
- behavior distance: общие тесты, общие трассы

#### Refinement != refactor

Refinement в коде: переход к более дорогому и локальному способу понять или изменить выбранный регион. Частные случаи: спуск на мелкие юниты, построение точного подграфа, запуск дорогого анализа, таргетные тесты, локальный рефакторинг, оптимизация, постановка guardrails.

#### Три слоя Curiosity-on-code

1. **Space representation:** tree / graph / hybrid
2. **Selection semantics:** rho(x) + payoff vs cost + probe
3. **Refinement action:** deeper analysis / verification / refactor / optimization / policy hardening

**Контекст:** анализ joi-lab/ouroboros (self-modifying AI agent) + уточнение от Агента 1.

**Статус:** заморожено до завершения Phase 4.

---

