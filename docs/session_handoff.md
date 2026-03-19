# Session Handoff — Curiosity Phase 1

Документ для новой сессии AI-оркестратора. Содержит полный контекст для немедленного продолжения работы.

## Где мы

Проект Curiosity. Фаза 0 **завершена** (18 марта 2026). Фаза 1 **завершена** (19 марта 2026). P0 layout **ЗАКРЫТ**. Следующий шаг — Фаза 2.

Рабочий ПК: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Рабочая директория: `R:\Projects\Curiosity`.

## Что читать (в этом порядке)

| # | Файл | Зачем |
|---|------|-------|
| 1 | `docs/phase1_plan.md` | **План Фазы 1** — 6 потоков, worker assignment, kill criteria, reuse map |
| 2 | `docs/concept_v1.8.md` | Каноническая концепция (актуальная) |
| 3 | `docs/experiment_hierarchy.md` | Граф зависимостей, приоритеты, нумерация exp10+ |
| 4 | `docs/teamplan.md` | План с отметкой Фаза 0, описание Фаз 1-4 |
| 5 | `docs/environment_2.md` | Как активировать .venv-gpu на PC 2 (CUDA) |

## Что было сделано ранее (Фаза 0)

### Эксперименты
1. **Окружение**: PC 1 (AMD Radeon 780M, DirectML) + PC 2 (RTX 2070, CUDA 12.8)
2. **Halo cross-space**: grid/graph OK, tree FAIL (0.56x). Правило: parallelism >= 3 AND no leakage.
3. **P2a sweep**: код готов (20K конфигураций), НЕ ЗАПУЩЕН
4. **SC-baseline**: D_parent = ||R(delta)|| / (||delta|| + eps), R = gauss sigma=3.0. AUC 0.824-1.000 на 4 пространствах.

### Ключевые архитектурные решения
- Halo: НЕ универсальный инвариант — правило по топологии.
- D_parent: формула обновлена (lf_frac).
- Morton/block-sparse/phase schedule: ОТЛОЖЕНЫ (deferred), не отвергнуты.

## Фаза 1 — Результаты основных потоков (19 марта 2026)

| Stream | Результат |
|--------|-----------|
| S1 exp10 | KILL compact-with-reverse-map (VRAM +38.6%). Grid — baseline. Убита реализация, не принцип sparse. |
| S1b exp10d | DET-1 PASS (240/240 побитовое совпадение CPU+CUDA) |
| S2 exp11 | FAIL 3/4 spaces. Архитектурная проблема. |
| S3 P2a | PASS — ridge 100%. Ручные пороги ok. P2b не нужен. |
| S4 exp12a | Thresholds найдены. L1 specificity низкая. |
| S5 deferred | Research note done. |

Gate Phase 1 -> 2: PASSED (grid fixed + DET-1). P0 переоткрыт для layout investigation.

---

## P0 LAYOUT — ЗАКРЫТ (полная серия exp10, 19 марта 2026)

### Словарь layout-ов

- **D_direct** ("packed tiles + прямой tile_map") — активные тайлы в компактном массиве, tile_map[tile_id] -> slot для O(1) lookup. Без element-level reverse_map. Победитель для сеток.
- **A_bitset** ("плотная сетка + bitset маска") — полноразмерный тензор данных + битовая маска активации. Простой fallback.
- **D_blocked** ("блочная адресация для графов") — узлы графа разбиты на фиксированные блоки, block_map[block_id] -> slot. Работает только для пространственных графов.
- **E_hash** ("hash table lookup") — архивный fallback, доминируется D_direct на текущем масштабе. Триггеры воскрешения задокументированы.

### Полная хронология серии exp10

| Эксперимент | Что проверяли | Результат |
|-------------|--------------|-----------|
| exp10 | Grid layout vs compact layout с element-level reverse_map (scalar_grid) | KILL compact. reverse_map[M] на int32 = структурный провал VRAM (+38.6%). Compute на O(k) был на 18.5% быстрее. |
| exp10d | Побитовый детерминизм DET-1 (все 4 типа пространств) | PASS 240/240 побитовое совпадение CPU+CUDA. |
| exp10e | Три tile-sparse кандидата: A=grid+bitset маска, B=packed tiles+Morton бинарный поиск, C=paged sparse (scalar_grid) | A жив (-20% время, +18% VRAM). B убит (бинарный поиск +1700%). C убит (+9000%). |
| exp10f | Packed tiles + прямой tile_map O(1) vs hash table (scalar_grid) | D_direct: 5x быстрее, 5.5x меньше resident. Peak VRAM kill = артефакт измерения. E_hash = та же скорость, build в 10-30x дольше. |
| exp10g | Двухрежимный бенчмарк: ручной stencil (Контур A) + conv2d (Контур B) (scalar_grid) | D_direct PASS оба контура. -54% до -80% время, -36% до -86% peak VRAM. |
| exp10h | Кросс-пространства: vector_grid + tree_hierarchy | Vector: 72/72 PASS оба контура. Tree: 0/108 FAIL (деревья слишком маленькие для амортизации overhead). |
| exp10i | Блочная адресация для графов с 3 стратегиями разбиения (3 типа графов) | Пространственные графы (random_geometric, grid_graph): условно жизнеспособны с spatial partition, cbr<0.30. Scale-free (barabasi-albert): ОТКЛОНЁН, cbr=0.66. |
| exp10j | Per-level анализ break-even D_direct для деревьев (tree_hierarchy) | matmul: D побеждает при occupancy < 37.5-40% на ВСЕХ размерах уровня. stencil: D экономит память, но НИКОГДА не выигрывает по времени. Контур B: 45% PASS. |

### Финальная layout policy

| Тип пространства | Layout | Статус | Доказательство |
|-----------------|--------|--------|---------------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production | exp10g: оба контура PASS |
| vector_grid | D_direct (packed tiles + tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Гибрид: D_direct per-level где occupancy < 40% + тяжёлый compute; A_bitset остальное | Validated | exp10j: break-even найден |
| irregular_graph / spatial | D_blocked (блочная адресация) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (плотная сетка + bitset маска) fallback | Только fallback | exp10i: блоки отклонены, cbr=0.66 |

**Break-even для деревьев (exp10j):**
- matmul оператор: D_direct побеждает при occupancy < 37.5-40% на ЛЮБОМ размере уровня
- stencil оператор: D_direct экономит память ниже того же порога, но всегда медленнее
- Policy: использовать D_direct per-level только когда оператор compute-heavy (matmul-like) И occupancy < 40%
- Верхние уровни дерева (маленький N_l, высокая occupancy) -> A_bitset
- Нижние уровни (большой N_l, низкая occupancy, тяжёлый compute) -> D_direct

### Убито навсегда

- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Бинарный поиск на GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash как основной lookup (exp10f-E: доминируется D_direct)
- Фиксированные блоки для scale-free графов (exp10i: cbr 0.64-0.99)

---

## Что делать — Фаза 2

P0 layout **ЗАКРЫТ**. Все типы пространств имеют назначенный layout (см. таблицу выше).

### Критический путь
```
P0 LAYOUT [CLOSED] -> Phase 2 (интеграция layout в runtime)
```

### Открытые вопросы для архитектора (перед началом Фазы 2)
- exp11: переделывать dirty signatures или принять ограничение? (FAIL 3/4 spaces)
- exp12a: L1 без enforcement или доработка? (specificity низкая)
- DET-2 cross-seed stability: нужен ли до начала Фазы 2?

### Возможные follow-up (НЕ в текущей фазе)
- Для графов: variable-size/adaptive blocks для spatial subclass; graph-native sparse (CSR/COO) для scale-free
- Для деревьев: stencil path оптимизация (текущий ручной stencil слишком медленный) — низкий приоритет
- Для всех: DET-2 cross-seed stability ещё не проверен

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
- **10-20 seeds** — для воспроизводимости
- **Barmagloth = архитектор** — принимает решения на развилках, не пишет код
