# Инвентарь артефактов из диалогов / Artifact Inventory

Составлен при ревью чатов Claude и ChatGPT проекта Curiosity.

---

## Claude (8 чатов в проекте)

### Чат 1: «Передача проекта Cloud Orchestrator'у» (4 days ago)
**Артефакты: нет** — текстовый handoff-диалог.

### Чат 2: «План экспериментов по адаптивной оптимизации» (19 days ago)
**Артефакты (1):**
- [ ] `Curiosity experiment hierarchy` (MD) — **уже в гите** как `docs/experiment_hierarchy.md`

### Чат 3: «Фаза 2: проверка слепоты через probe-стратегии» (19 days ago)
**Артефакты (15) — ✅ ВСЕ СКАЧАНО:**
- [x] `Phase2 full report` (MD) → `experiments/phase2/phase2_full_report.md`
- [x] `Exp phase2 probe` (PY) → `experiments/phase2/exp_phase2_probe.py`
- [x] `Phase2 probe` (PNG) → `experiments/phase2/phase2_probe.png`
- [x] `Phase2 quiet timeline` (PNG) → `experiments/phase2/phase2_quiet_timeline.png`
- [x] `Phase2 probe` (JSON) → `experiments/phase2/phase2_probe.json`
- [x] `Exp seam metric` (PY) → `experiments/phase2/exp_seam_metric.py`
- [x] `Phase2 seam metric` (PNG) → `experiments/phase2/phase2_seam_metric.png`
- [x] `Phase2 seam metric` (JSON) → `experiments/phase2/phase2_seam_metric.json`
- [x] `Exp seam metric v2` (PY) → `experiments/phase2/exp_seam_metric_v2.py`
- [x] `Seam metric v2` (PNG) → `experiments/phase2/seam_metric_v2.png`
- [x] `Seam metric v2` (JSON) → `experiments/phase2/seam_metric_v2.json`
- [x] `Exp seam crossspace` (PY) → `experiments/phase2/exp_seam_crossspace.py`
- [x] `Seam crossspace` (PNG) → `experiments/phase2/seam_crossspace.png`
- [x] `Seam crossspace` (JSON) → `experiments/phase2/seam_crossspace.json`
- [x] `Phase2 summary report` (MD) → `experiments/phase2/phase2_summary_report.md`

### Чат 4: «Выбор структуры данных: Morton vs динамический список» (23 days ago)
**Артефакты — ✅ ВСЕ СКАЧАНО (2 superseded отброшены):**
- [x] `Phase1 summary report final` (MD) → `experiments/phase1/phase1_summary_report_final.md`
- [x] `Phase1 A1 final` (PNG) → `experiments/phase1/phase1_A1_final.png`
- [x] `Phase1 a2ext v2` (PNG) → `experiments/phase1/phase1_a2ext_v2.png`
- [x] `Phase1 a2ext curves` (PNG) → `experiments/phase1/phase1_a2ext_curves.png`
- [x] `Phase1 a2ext rmin vs spectral` (PNG) → `experiments/phase1/phase1_a2ext_rmin_vs_spectral.png`
- [x] `Phase1 analytical vs empirical` (PNG) → `experiments/phase1/phase1_analytical_vs_empirical.png`
- [x] `Phase1 hardened` (PNG) → `experiments/phase1/phase1_hardened.png`
- [x] `Phase1 v3 spatial` (PNG) → `experiments/phase1/phase1_v3_spatial.png`
- [x] `Phase1 v3 summary` (PNG) → `experiments/phase1/phase1_v3_summary.png`
- [x] `Exp phase1 hardened` (PY) → `experiments/phase1/exp_phase1_hardened.py`
- [x] `Exp phase1 a2ext v2` (PY) → `experiments/phase1/exp_phase1_a2ext_v2.py`
- [x] `Exp phase1 v3` (PY) → `experiments/phase1/exp_phase1_v3.py`
- [x] `Phase1 hardened` (JSON) → `experiments/phase1/phase1_hardened.json`
- [x] `Phase1 v3 results` (JSON) → `experiments/phase1/phase1_v3_results.json`
- ~~`Exp phase1 a2ext` (PY)~~ — superseded by v2 (порог 5% → knee detection)
- ~~`Phase1 summary report` (MD)~~ — superseded by _final (hardened)

### Чат 5: «Нужен ли динамический schedule для управления бюджетом» (24 days ago)
**Артефакты — ✅ СКАЧАНО (3 файла в experiments/exp08/, PNG ожидают):**
- [x] `Exp08v5 schedule` (PY) → `experiments/exp08/exp08v5_schedule.py`
- [x] `Exp08 design` (MD) → `experiments/exp08/exp08_design.md`
- [ ] `Exp08 clean` (PNG) — график clean ⚠️ ожидает ручной download
- [ ] `Exp08 shift` (PNG) — график shift ⚠️ ожидает ручной download
- [x] `Exp08v5 summary` (JSON) → `experiments/exp08/exp08v5_summary.json`
- ~~`Curiosity concept v1.5` (MD)~~ — уже в гите, идентичен (до pixel→cell правок)

### Чат 6: «Комбинированная интересность в адаптивной иерархии» (25 days ago)
**Артефакты (7) — НУЖНО СКАЧАТЬ:**
- [ ] `Exp04 protocol` (MD) — протокол Exp0.4
- [ ] `Exp04 combined interest` (PY) — скрипт Exp0.4
- [ ] `Run metrics` (JSON) — данные Exp0.4
- [ ] `Exp05 break oracle` (PY) — скрипт Exp0.5
- [ ] `Metrics noise` (JSON) — данные noise
- [ ] `Exp07b twostage` (PY) — скрипт Exp0.7b
- [ ] `Exp07 protocol` (MD) — протокол Exp0.7

### Чат 7: «Adaptive algorithm performance evaluation» (26 days ago)
**Артефакты (1) — НУЖНО СКАЧАТЬ:**
- [ ] `Exp03 halo diagnostic` (IPYNB) — Jupyter notebook Exp0.3

### Чат 8: «Проверка гипотезы Adaptive Refinement» (26 days ago)
**Артефакты: нет** — ответ был interrupted, файл потерян.

---

## ChatGPT (9 чатов в проекте)

### Sources (файлы проекта):
- `curiosity_concept_v1.5.md` — **уже в гите**
- `Концептуальная фиксация (v14).txt` — **уже в гите** как `concept_v1.4_historical.md`
- `curiosity_план_работ.md` — **уже в гите** как `workplan.md`

### Чаты:
1. **Phase 2-3 эксперимент** (Feb 19) — текст, рабочий порядок экспериментов. Артефактов не обнаружено.
2. **Фаза 2 Старт** (Feb 15) — не проверен детально, требует ручного ревью.
3. **Метрика шва в латентном пространстве** (Feb 15) — не проверен, возможны canvas-артефакты с кодом.
4. **Закрепление плана работ** (Feb 15) — не проверен.
5. **New chat** (Feb 14) — не проверен.
6. **Exp0.8 schedule issue** (Feb 14) — не проверен, вероятны canvas-артефакты с кодом Exp0.8.
7. **Комбинированная интересность ρ** (Feb 14) — не проверен.
8. **Техзадание для кода** (Feb 12) — проверен, текстовый, без артефактов.
9. **План работы пункта А** (Feb 11) — не проверен.

---

## Уже в репозитории

| Файл | Источник |
|------|----------|
| experiments/exp09a/ (3 файла) | Загружены пользователем |
| experiments/phase2/ (15 файлов) | Claude чат 3 (✅ полностью) |
| experiments/phase1/ (14 файлов) | Claude чат 4 (✅ полностью, 2 superseded отброшены) |
| experiments/exp08/ (3 файла) | Claude чат 5 (скачано, PNG ожидают) |
| docs/concept_v1.5.md | Claude + ChatGPT |
| docs/concept_v1.4_historical.md | ChatGPT |
| docs/workplan.md | ChatGPT |
| docs/experiment_hierarchy.md | Claude чат 2 |

---

## Приоритет скачивания

**Высокий (код + данные экспериментов):**
1. ~~Чат 3 Claude~~ — ✅ done
2. ~~Чат 4 Claude~~ — ✅ done
3. Чат 6 Claude — 7 файлов Exp0.4 + Exp0.5 + Exp0.7
4. ~~Чат 5 Claude~~ — ✅ done (PNG ожидают)
5. Чат 7 Claude — 1 notebook Exp0.3

**Средний (ChatGPT чаты — возможны canvas-артефакты):**
6. ChatGPT «Exp0.8 schedule issue» — вероятен код
7. ChatGPT «Метрика шва в латентном пространстве» — вероятен код
8. ChatGPT «Комбинированная интересность ρ» — вероятен код

**Низкий (текстовые обсуждения):**
9. Остальные ChatGPT чаты
