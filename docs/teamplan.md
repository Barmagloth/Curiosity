# Curiosity: Параллельный план работы команды

## Контекст

Проект Curiosity завершил серию Exp0.1–0.8. Впереди P0–P4 + SC-baseline + кросс-пространственная валидация Halo. Задача — распределить работу на 4+ параллельных исполнителя при условии, что Barmagloth выступает архитектором (принимает решения на развилках, ревьюит результаты, не пишет код).

**Ограничение**: GPU — AMD Radeon 780M (нет CUDA). Используется DirectML + PyTorch (Python 3.12). CPU venv на Python 3.13.

---

## Потоки работы (Streams)

### ✅ Фаза 0: ЗАВЕРШЕНА (18 марта 2026)

Четыре потока, все независимы друг от друга:

| Поток | Исполнитель | Задача | Тип | Зависимости |
|-------|-------------|--------|-----|-------------|
| **S1: Окружение** | Executor A | Поднять ROCm + PyTorch на AMD GPU. Проверить совместимость существующего кода (exp07, exp08). Написать `environment_1.md` с версиями. | Инфраструктура | Нет — стартует сразу |
| **S2: Halo cross-space** | Executor B | Расширить `phase2_probe_seam/exp_seam_crossspace.py` — валидировать Halo (cosine feathering, ≥3 элемента) на всех 4 типах пространств (scalar grid, vector grid, irregular graph, tree hierarchy). CPU-only. | Валидация | Нет — переиспользует код phase2 |
| **S3: P2a sweep design** | Executor C | Реализовать sensitivity sweep порогов (instability_threshold, FSR_threshold) по 5 сценам (clean/noise/blur/spatvar/jpeg) **× 4 типа пространств** (scalar grid, vector grid, irregular graph, tree hierarchy). CPU-only, переиспользует код exp07/exp08 + phase2 cross-space инфраструктуру. | Эксперимент | Нет — данные и код есть |
| **S4: SC-baseline scaffold** | Executor D | Реализовать каркас SC-baseline по протоколу `scale_consistency_verification_protocol_v1.0.md`: SC-0 (idempotence R), SC-1 (positive/negative baselines), SC-2 (D_parent, D_hf вычисление). CPU-only. | Код + валидация | Нет — протокол готов |

**Результаты Фазы 0:**
- S1: CPU venv (Python 3.13) + GPU venv (Python 3.12 + DirectML, Radeon 780M, 2-3× speedup at 2048+)
- S2: Halo FAIL на деревьях (0.56×). Правило: parallelism ≥ 3 AND no context leakage. Grid/graph: ok. Tree: never.
- S3: P2a sweep код готов (20K конфигураций), ещё не запущен
- S4: SC-baseline пройден. D_parent обновлён: ‖R(δ)‖ / (‖δ‖ + ε), R=σ3.0. AUC 0.824–1.000 на 4 пространствах.

**Дополнительно выполнено (вне плана):**
- coarse_shift генератор исправлен (spatially coherent sign fields)
- D_parent combo sweep: 66 комбинаций протестировано
- abs_vs_signed auxiliary отвергнут (контрпродуктивен с исправленным генератором)

**Развилки для архитектора (конец Фазы 0):** ✅ Все решены
- S2 результат: Halo работает на всех 4 пространствах → статус «обязательный инвариант» подтверждён. Если нет → решение о модификации Halo.
- S1 результат: ROCm работает → переход к GPU-экспериментам. Если нет → fallback на облако / WSL + CUDA.

---

### ✅ Фаза 1: ЗАВЕРШЕНА (20 марта 2026)

Все потоки PASS. P0 Layout закрыт. DET-1 и DET-2 пройдены. Gate Phase 1 -> Phase 2: PASSED.

#### Оригинальный план (Неделя 3–5)

| Поток | Исполнитель | Задача | Тип | Зависимости |
|-------|-------------|--------|-----|-------------|
| **S1: P0 — Exp0.9b0** | Executor A | Buffer-scaling probe: O(k) vs O(M) overhead. Grid vs compact. Kill compact если overhead > 20%. Требует GPU. | Эксперимент (GPU) | S1 Фаза 0 (окружение) |
| **S1b: DET-1** | Executor A | Seed determinism: canonical traversal order (Z-order tie-break), deterministic probe, governor isolation. Поглощает 0.9h. Kill: любое расхождение = fail. **Блокер для Фазы 2.** | Валидация | S1 P0 (layout определяет порядок обхода) |
| **S2: P1-B2 прототип** | Executor B | Dirty signatures: 12-bit signature (seam_risk + uncert + mass), debounce, AUC > 0.8. CPU-прототип, позже перенос на GPU. | Код + эксперимент | Нет (CPU) |
| **S3: P2a выполнение** | Executor C | Запуск sensitivity sweep (из Фазы 0) на 5 сцен × 4 пространства. Определить: ridge width > 30% → ручные пороги ок; < 10% → нужен P2b. **Важно:** если ridge width различается между пространствами — это само по себе значимый результат, требующий решения архитектора. | Эксперимент | S3 Фаза 0 (код готов) |
| **S4: SC-baseline завершение** | Executor D | SC-0..SC-4 пройдены. Осталось: SC-5 — установить data-driven τ_parent[L]. Подготовить SC-enforce (Фаза 2). | Валидация | S4 Фаза 0 (каркас) ✅ |
| **S5: Deferred revisit** | Executor E | Re-investigation Morton layout / block-sparse / phase schedule с иным подходом. Литобзор + новые идеи. Не эксперимент — research note с предложениями. | Исследование | Нет |

**Gate: Фаза 1 → Фаза 2: ✅ PASSED**
- P0 layout закрыт — полная layout policy по типам пространств.
- DET-1 PASS (240/240 побитовое совпадение CPU+CUDA).
- DET-2 PASS (cross-seed stability).
- Все потоки (S1–S5, exp11, exp12a) — PASS.

**Развилки для архитектора (конец Фазы 1): ✅ Все решены**
- Layout: D_direct для сеток (scalar + vector), гибрид для деревьев, D_blocked conditional для spatial графов, A_bitset fallback для scale-free.
- P2b нужен? Нет — ridge 100%.
- SC-5: пороги найдены.
- Morton/block-sparse: Morton убит (бинарный поиск +1700%), block addressing viable только для spatial графов.

---

### ✅ Фаза 2: End-to-end pipeline validation (Неделя 6–8) — ЗАВЕРШЕНА (21 марта 2026)

| Поток | Исполнитель | Задача | Зависимости |
|-------|-------------|--------|-------------|
| **S1: P0 завершение + DET-2** | Executor A | Exp0.9b (end-to-end если compact жив). DET-2 (cross-seed stability, 20 seeds × 4 пространства × 2 бюджета). | P0 + DET-1 (Фаза 1) |
| **S2: P1-B1 compression** | Executor B | Segment compression (degree-2 + signature-stable + length cap). Compression ratio > 50%, overhead < 10%. | P1-B2 (Фаза 1) + P0 layout + DET-1 |
| **S3: P2b (условно)** | Executor C | Online percentile estimation для adaptive threshold. Только если P2a показал narrow ridge. Иначе — помогает другим потокам. | P2a результат |
| **S4: SC-enforce** | Executor D | Damp delta / reject split при D_parent > τ_parent. Интеграция enforcement в pipeline. | SC-baseline pass |
| **S5: Enox infra** | — | Четыре observation-only паттерна (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep). Чистая аннотация, zero functional change. Заготовка для Phase 3. | Phase 2 pipeline |

**Результаты Фазы 2:**
- S1 (Pipeline Assembly): CuriosityPipeline собран — gate + governor + SC-enforce + probe + traversal. ✅ DONE
- S2 (SC-Enforce): Three-tier pass/damp/reject + strictness-weighted waste budget + adaptive τ T4(N). ✅ DONE
- S3 (Segment Compression): Thermodynamic guards (N_critical=12, bombardment). Compression 60-66% on d7/d8. ✅ DONE
- S4 (E2E Validation): 240 configs, 4 spaces, DET-1 40/40 + DET-2 8/8. Topo profiling integrated. ✅ DONE
- S5 (Enox Infra): 4 observation-only patterns. ✅ DONE. Comparison: NO REGRESSION (15/20 bitwise SAME). DET-1 PASS.

**Gate: Фаза 2 → Фаза 3: ✅ PASSED**
- Pipeline собран и E2E валидирован
- SC-enforce интегрирован
- Topo profiling интегрирован
- Enox-инфраструктура: ✅ DONE. NO REGRESSION

**Развилки для архитектора (конец Фазы 2): ✅ Все решены**
- SC-enforce работает: три уровня (pass/damp/reject), adaptive τ для деревьев
- Compression: прибыльна на d7/d8, guards отсекают невыгодные случаи
- Enox: ADOPT RegionURI hash, ADAPT decision journal/dedup/sweep/provenance, SKIP phase sep/perspectives

---

### Фаза 3: Семантика + rebuild (Неделя 9–11)

| Поток | Исполнитель | Задача | Зависимости |
|-------|-------------|--------|-------------|
| **S1: P1-B3 anchors** | Executor A/B | Periodic rebuild + anchor insertion. Divergence < 5% vs full rebuild. | P1-B1 |
| **S2: P3a LCA-distance** | Executor C | Корреляция LCA-distance с feature similarity. Correlation > 0.3 → дерево семантично. | P1-B1 (compressed tree) |
| **S3: P3b bushes** | Executor D | Кластеры leaf-paths. Silhouette > 0.4 + стабильность. | P3a (можно параллельно) |

---

### Фаза 4: Интеграция (Неделя 12–14)

| Поток | Исполнитель | Задача | Зависимости |
|-------|-------------|--------|-------------|
| **S1: P4a downstream** | Executor A | Classifier/autoencoder на adaptive-refined vs dense vs coarse. Metric loss < 2%. | Все P0–P3 + SC |
| **S2: P4b matryoshka** | Executor B | Каждый уровень вложенности валиден для downstream. | P4a |
| **S3: C-pre** | Executor C | Есть ли natural clustering в trajectory features? Go/no-go для Track C. | P3 результаты |

---

## Критический путь

```
Фаза 0: S1(env) ──→ Фаза 1: S1(P0) → S1b(DET-1) ──→ Фаза 2: S2(P1-B1) ──→ Фаза 3: S1(P1-B3) ──→ Фаза 4: S1(P4)
                                                   └──→ Фаза 2: S1(P0 finish + DET-2)
```

**Критический путь:** Env → P0 → DET-1 → P1(B2→B1→B3) → P4 = ~14 недель

**Параллельные потоки сокращают реальное время:**
- Halo cross-space (Фаза 0) — не на критическом пути, но блокирует уверенность в инвариантах
- P2, SC-baseline — параллельны с P1, не удлиняют критический путь
- DET-2 — параллельна с P1-B1 (после DET-1), блокирует Track B но не Фазу 3
- Deferred revisit — чистый research, не блокирует ничего

---

## Точки принятия решений архитектором

| Когда | Что решать | Входные данные |
|-------|-----------|---------------|
| Конец Фазы 0 | ✅ Решено | Результат |
|---|---|---|
| ROCm? | CPU + DirectML (ROCm не поддерживает Windows iGPU) |
| Halo кросс-пространственный? | Частично: grid/graph да, tree нет. Правило выведено. |
| D_parent fail? | Исправлен: σ=3.0 + lf_frac normalization |
| coarse_shift? | Генератор исправлен на spatially coherent |
| Конец Фазы 1 | ✅ Решено (20 марта 2026). Layout: D_direct/гибрид/D_blocked/A_bitset по типам. P2b не нужен. SC pass. Morton убит. | Все потоки PASS |
| Конец Фазы 2 | ✅ Решено (21 марта 2026). SC-enforce: pass/damp/reject. Compression: 60-66%. Enox: 4 observation-only patterns. | Все потоки PASS |
| Конец Фазы 3 | Дерево семантично? C unfreezes? | P3a, P3b отчёты |
| Конец Фазы 4 | Instrument Readiness Gate пройден? Переход на Track B? | P4a, P4b, C-pre |

---

## Ключевые файлы

- `docs/experiment_hierarchy.md` — граф зависимостей, kill criteria
- `docs/workplan.md` — модули A–F
- `docs/scale_consistency_verification_protocol_v1.0.md` — протокол SC
- `docs/concept_v1.8.md` — каноническая концепция (актуальная)
- `experiments/phase2_probe_seam/` — код для переиспользования в Halo cross-space
- `experiments/exp07_gate/`, `experiments/exp08_schedule/` — код для P2a sweep

---

## Верификация

После каждой фазы:
1. Все kill criteria проверены (числа, не мнения)
2. Holm-Bonferroni коррекции при множественных сравнениях
3. 10–20 seeds для воспроизводимости
4. Результаты записаны в `docs/experiment_results.md` (append)
5. Архитектор ревьюит перед переходом к следующей фазе

**Принцип кросс-пространственной валидации:** Любой эксперимент, претендующий на утверждение об «произвольных пространствах», ОБЯЗАН проверяться минимум на 4 типах пространств (scalar grid, vector grid, irregular graph, tree hierarchy) — тех же, на которых валидирован SeamScore. Результат на одном типе пространства (например, 2D pixel grid) НЕ считается достаточным для обобщения.
