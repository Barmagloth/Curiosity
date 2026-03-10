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
- [x] `Phase2 full report` (MD) → `experiments/phase2_probe_seam/phase2_full_report.md`
- [x] `Exp phase2 probe` (PY) → `experiments/phase2_probe_seam/exp_phase2_probe.py`
- [x] `Phase2 probe` (PNG) → `experiments/phase2_probe_seam/phase2_probe.png`
- [x] `Phase2 quiet timeline` (PNG) → `experiments/phase2_probe_seam/phase2_quiet_timeline.png`
- [x] `Phase2 probe` (JSON) → `experiments/phase2_probe_seam/phase2_probe.json`
- [x] `Exp seam metric` (PY) → `experiments/phase2_probe_seam/exp_seam_metric.py`
- [x] `Phase2 seam metric` (PNG) → `experiments/phase2_probe_seam/phase2_seam_metric.png`
- [x] `Phase2 seam metric` (JSON) → `experiments/phase2_probe_seam/phase2_seam_metric.json`
- [x] `Exp seam metric v2` (PY) → `experiments/phase2_probe_seam/exp_seam_metric_v2.py`
- [x] `Seam metric v2` (PNG) → `experiments/phase2_probe_seam/seam_metric_v2.png`
- [x] `Seam metric v2` (JSON) → `experiments/phase2_probe_seam/seam_metric_v2.json`
- [x] `Exp seam crossspace` (PY) → `experiments/phase2_probe_seam/exp_seam_crossspace.py`
- [x] `Seam crossspace` (PNG) → `experiments/phase2_probe_seam/seam_crossspace.png`
- [x] `Seam crossspace` (JSON) → `experiments/phase2_probe_seam/seam_crossspace.json`
- [x] `Phase2 summary report` (MD) → `experiments/phase2_probe_seam/phase2_summary_report.md`

### Чат 4: «Выбор структуры данных: Morton vs динамический список» (23 days ago)
**Артефакты — ✅ ВСЕ СКАЧАНО (2 superseded отброшены):**
- [x] `Phase1 summary report final` (MD) → `experiments/phase1_halo/phase1_summary_report_final.md`
- [x] `Phase1 A1 final` (PNG) → `experiments/phase1_halo/phase1_A1_final.png`
- [x] `Phase1 a2ext v2` (PNG) → `experiments/phase1_halo/phase1_a2ext_v2.png`
- [x] `Phase1 a2ext curves` (PNG) → `experiments/phase1_halo/phase1_a2ext_curves.png`
- [x] `Phase1 a2ext rmin vs spectral` (PNG) → `experiments/phase1_halo/phase1_a2ext_rmin_vs_spectral.png`
- [x] `Phase1 analytical vs empirical` (PNG) → `experiments/phase1_halo/phase1_analytical_vs_empirical.png`
- [x] `Phase1 hardened` (PNG) → `experiments/phase1_halo/phase1_hardened.png`
- [x] `Phase1 v3 spatial` (PNG) → `experiments/phase1_halo/phase1_v3_spatial.png`
- [x] `Phase1 v3 summary` (PNG) → `experiments/phase1_halo/phase1_v3_summary.png`
- [x] `Exp phase1 hardened` (PY) → `experiments/phase1_halo/exp_phase1_hardened.py`
- [x] `Exp phase1 a2ext v2` (PY) → `experiments/phase1_halo/exp_phase1_a2ext_v2.py`
- [x] `Exp phase1 v3` (PY) → `experiments/phase1_halo/exp_phase1_v3.py`
- [x] `Phase1 hardened` (JSON) → `experiments/phase1_halo/phase1_hardened.json`
- [x] `Phase1 v3 results` (JSON) → `experiments/phase1_halo/phase1_v3_results.json`
- ~~`Exp phase1 a2ext` (PY)~~ — superseded by v2 (порог 5% → knee detection)
- ~~`Phase1 summary report` (MD)~~ — superseded by _final (hardened)

### Чат 5: «Нужен ли динамический schedule для управления бюджетом» (24 days ago)
**Артефакты — ✅ ВСЕ СКАЧАНО:**
- [x] `Exp08v5 schedule` (PY) → `experiments/exp08_schedule/exp08v5_schedule.py`
- [x] `Exp08 design` (MD) → `experiments/exp08_schedule/exp08_design.md`
- [x] `Exp08 clean` (PNG) → `experiments/exp08_schedule/exp08_clean.png`
- [x] `Exp08 shift` (PNG) → `experiments/exp08_schedule/exp08_shift.png`
- [x] `Exp08 noise` (PNG) → `experiments/exp08_schedule/exp08_noise.png`
- [x] `Exp08 spatvar` (PNG) → `experiments/exp08_schedule/exp08_spatvar.png`
- [x] `Probe ablation` (PNG) → `experiments/exp08_schedule/probe_ablation.png`
- [x] `Exp08v5 summary` (JSON) → `experiments/exp08_schedule/exp08v5_summary.json`
- ~~`Curiosity concept v1.5` (MD)~~ — уже в гите, идентичен (до pixel→cell правок)

### Чат 6: «Комбинированная интересность в адаптивной иерархии» (25 days ago)
**Артефакты (13) — ✅ ВСЕ СКАЧАНО (по 4 подпапкам exp04–exp07):**
- [x] `Exp04 protocol` (MD) → `experiments/exp04_combined_interest/exp04_protocol.md`
- [x] `Exp04 combined interest` (PY) → `experiments/exp04_combined_interest/exp04_combined_interest.py`
- [x] `Run metrics` (JSON) → `experiments/exp04_combined_interest/run_metrics.json`
- [x] `Metrics clean` (JSON) → `experiments/exp04_combined_interest/metrics_clean.json`
- [x] `Metrics alias` (JSON) → `experiments/exp04_combined_interest/metrics_alias.json`
- [x] `Metrics blur` (JSON) → `experiments/exp04_combined_interest/metrics_blur.json`
- [x] `Exp05 break oracle` (PY) → `experiments/exp05_break_oracle/exp05_break_oracle.py`
- [x] `Metrics noise` (JSON) → `experiments/exp05_break_oracle/metrics_noise.json`
- [x] `Exp06 adaptive` (PY) → `experiments/exp06_adaptive_switch/exp06_adaptive.py`
- [x] `Exp07 protocol` (MD) → `experiments/exp07_gate/exp07_protocol.md`
- [x] `Exp07 soft gate` (PY) → `experiments/exp07_gate/exp07_soft_gate.py`
- [x] `Exp07b twostage` (PY) → `experiments/exp07_gate/exp07b_twostage.py`
- [x] `Exp07 summary` (JSON) → `experiments/exp07_gate/exp07_summary.json`

### Чат 7: «Adaptive algorithm performance evaluation» (26 days ago)
**Артефакты (3) — ✅ ВСЕ СКАЧАНО:**
- [x] `Exp01 adaptive refinement` (IPYNB) → `experiments/exp01_poc/exp01_adaptive_refinement.ipynb`
- [x] `Exp02 cifar adaptive refinement` (IPYNB) → `experiments/exp02_cifar_poc/exp02_cifar_adaptive_refinement.ipynb`
- [x] `Exp03 halo diagnostic` (IPYNB) → `experiments/exp03_halo_diagnostic/exp03_halo_diagnostic.ipynb`

### Чат 8: «Проверка гипотезы Adaptive Refinement» (26 days ago)
**Артефакты: нет** — ответ был interrupted, файл потерян.

---

## ChatGPT (9 чатов в проекте)

### Sources (файлы проекта):
- `curiosity_concept_v1.5.md` — **уже в гите**
- `Концептуальная фиксация (v14).txt` — **уже в гите** как `concept_v1.4_historical.md`
- `curiosity_план_работ.md` — **уже в гите** как `workplan.md`

### Чаты:
Отложены — извлечение артефактов из ChatGPT требует ручного парсинга диалогов. TODO на потом.

---

## Уже в репозитории

| Файл | Источник |
|------|----------|
| experiments/exp09a_layout_sandbox/ (3 файла) | Загружены пользователем |
| experiments/phase2_probe_seam/ (15 файлов) | Claude чат 3 (✅ полностью) |
| experiments/phase1_halo/ (14 файлов) | Claude чат 4 (✅ полностью, 2 superseded отброшены) |
| experiments/exp08_schedule/ (8 файлов) | Claude чат 5 (✅ полностью) |
| experiments/exp04_combined_interest/ (6 файлов) | Claude чат 6 |
| experiments/exp05_break_oracle/ (2 файла) | Claude чат 6 |
| experiments/exp06_adaptive_switch/ (1 файл) | Claude чат 6 |
| experiments/exp07_gate/ (4 файла) | Claude чат 6 |
| experiments/exp01_poc/ (1 IPYNB) | Claude чат 7 |
| experiments/exp02_cifar_poc/ (1 IPYNB) | Claude чат 7 |
| experiments/exp03_halo_diagnostic/ (1 IPYNB) | Claude чат 7 |
| docs/concept_v1.5.md | Claude + ChatGPT |
| docs/concept_v1.4_historical.md | ChatGPT |
| docs/workplan.md | ChatGPT |
| docs/experiment_hierarchy.md | Claude чат 2 |

---

## Приоритет скачивания

**Высокий (код + данные экспериментов):**
1. ~~Чат 3 Claude~~ — ✅ done
2. ~~Чат 4 Claude~~ — ✅ done
3. ~~Чат 6 Claude~~ — ✅ done
4. ~~Чат 5 Claude~~ — ✅ done
5. ~~Чат 7 Claude~~ — ✅ done

**Claude чаты — ✅ ВСЕ ГОТОВО**

**ChatGPT чаты — отложены** (требуют ручного парсинга диалогов)
