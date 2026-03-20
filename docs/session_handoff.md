# Session Handoff — Curiosity Phase 2

Документ для новой сессии AI-оркестратора. Содержит полный контекст для немедленного продолжения работы.

## Где мы

Проект Curiosity.

- Фаза 0 **завершена** (18 марта 2026).
- Фаза 1 **завершена** (20 марта 2026). Все потоки — PASS. P0 Layout **ЗАКРЫТ**. DET-1 **PASS**. DET-2 **PASS**.
- **Следующий шаг — Фаза 2.**

Рабочий ПК: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Рабочая директория: `R:\Projects\Curiosity`.

## Что читать (в этом порядке)

| # | Файл | Зачем |
|---|------|-------|
| 1 | `docs/session_handoff.md` | **Этот файл** — текущий статус, что делать дальше |
| 2 | `docs/concept_v1.8.md` | Каноническая концепция (актуальная) |
| 3 | `docs/teamplan.md` | План с отметками Фаза 0, Фаза 1, описание Фаз 2-4 |
| 4 | `docs/experiment_hierarchy.md` | Граф зависимостей, приоритеты, нумерация exp10+ |
| 5 | `docs/environment_2.md` | Как активировать .venv-gpu на PC 2 (CUDA) |
| 6 | `docs/phase1_plan.md` | Архив — план и результаты Фазы 1 (справочно) |

## Фаза 1 — Финальные результаты (20 марта 2026)

### Основные потоки

| Stream | Результат | Статус |
|--------|-----------|--------|
| S1 exp10 | KILL compact-with-reverse-map (VRAM +38.6%). Grid — baseline. | PASS |
| S1b exp10d | DET-1 PASS (240/240 побитовое совпадение CPU+CUDA) | PASS |
| S2 exp11 | PASS | PASS |
| S3 P2a | PASS — ridge 100%. Ручные пороги ok. P2b не нужен. | PASS |
| S4 exp12a | PASS | PASS |
| S5 deferred | Research note done. | PASS |
| DET-2 | PASS (cross-seed stability) | PASS |

**Gate Phase 1 -> Phase 2: PASSED.** Все потоки PASS. P0 Layout закрыт. DET-1 и DET-2 пройдены.

---

## P0 LAYOUT — ЗАКРЫТ (полная серия exp10, 19 марта 2026)

### Словарь layout-ов

- **D_direct** ("packed tiles + прямой tile_map") — активные тайлы в компактном массиве, tile_map[tile_id] -> slot для O(1) lookup. Без element-level reverse_map. Победитель для сеток.
- **A_bitset** ("плотная сетка + bitset маска") — полноразмерный тензор данных + битовая маска активации. Простой fallback.
- **D_blocked** ("блочная адресация для графов") — узлы графа разбиты на фиксированные блоки, block_map[block_id] -> slot. Работает только для пространственных графов.
- **E_hash** ("hash table lookup") — архивный fallback, доминируется D_direct на текущем масштабе. Триггеры воскрешения задокументированы.

### Финальная layout policy

| Тип пространства | Layout | Статус | Доказательство |
|-----------------|--------|--------|---------------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production | exp10g: оба контура PASS |
| vector_grid | D_direct (packed tiles + tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Гибрид: D_direct per-level где occupancy < 40% + тяжёлый compute; A_bitset остальное | Validated | exp10j: break-even найден |
| irregular_graph / spatial | D_blocked (блочная адресация) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (плотная сетка + bitset маска) fallback | Только fallback | exp10i: блоки отклонены, cbr=0.66 |

### Убито навсегда

- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Бинарный поиск на GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash как основной lookup (exp10f-E: доминируется D_direct)
- Фиксированные блоки для scale-free графов (exp10i: cbr 0.64-0.99)

---

## Что делать — Фаза 2

### Цель

End-to-end pipeline validation. Собрать весь pipeline (layout + halo + gate + governor + probe + SeamScore) и прогнать на реальных задачах.

### Задачи

1. **Интеграция pipeline** — собрать все валидированные компоненты в единый runtime:
   - Layout (D_direct / A_bitset / D_blocked / гибрид) по типу пространства
   - Halo (cosine feathering, overlap >= 3)
   - Двухстадийный гейт (residual-first + utility-weighted fallback)
   - Budget governor (EMA-контроллер strictness)
   - Probe (5-10% бюджета на exploration)
   - SeamScore (Jumpout / (Jumpin + eps))

2. **End-to-end валидация** — прогнать собранный pipeline на реальных задачах, все 4 типа пространств

3. **SC-enforce** — интеграция enforcement (damp delta / reject split при D_parent > tau_parent)

### Критический путь

```
Phase 2 → Instrument Readiness Gate → Track A
```

---

## Открытые вопросы для Фазы 2

1. **Governor гистерезис для irregular high budget** — Track B. Как настроить EMA-контроллер для пространств с нерегулярным бюджетным профилем.

2. **Graph-native sparse для scale-free** — Track C. CSR/COO вместо A_bitset fallback для scale-free графов (barabasi-albert и подобных).

3. **C(I,M,p) surface с правильными метриками** — Track C. Построение поверхности curiosity = f(information, mass, position) с корректными метриками для каждого типа пространства.

---

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

---

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
