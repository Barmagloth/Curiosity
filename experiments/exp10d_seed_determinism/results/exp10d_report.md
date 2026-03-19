# exp10d — Seed Determinism Report

**Verdict:** PASS — bitwise determinism verified
**Total tests:** 240
**Pass:** 240
**Fail:** 0

## Configuration

- Seeds: 10 (0..9)
- Budgets: [('low', 0.1), ('medium', 0.3), ('high', 0.6)]
- Spaces: ['scalar_grid', 'vector_grid', 'irregular_graph', 'tree_hierarchy']
- Devices: ['cpu', 'cuda']

## Results by space x device

| Space | Device | Budget | Pass | Fail |
|-------|--------|--------|------|------|
| scalar_grid | cpu | low | 10/10 | 0/10 |
| scalar_grid | cpu | medium | 10/10 | 0/10 |
| scalar_grid | cpu | high | 10/10 | 0/10 |
| scalar_grid | cuda | low | 10/10 | 0/10 |
| scalar_grid | cuda | medium | 10/10 | 0/10 |
| scalar_grid | cuda | high | 10/10 | 0/10 |
| vector_grid | cpu | low | 10/10 | 0/10 |
| vector_grid | cpu | medium | 10/10 | 0/10 |
| vector_grid | cpu | high | 10/10 | 0/10 |
| vector_grid | cuda | low | 10/10 | 0/10 |
| vector_grid | cuda | medium | 10/10 | 0/10 |
| vector_grid | cuda | high | 10/10 | 0/10 |
| irregular_graph | cpu | low | 10/10 | 0/10 |
| irregular_graph | cpu | medium | 10/10 | 0/10 |
| irregular_graph | cpu | high | 10/10 | 0/10 |
| irregular_graph | cuda | low | 10/10 | 0/10 |
| irregular_graph | cuda | medium | 10/10 | 0/10 |
| irregular_graph | cuda | high | 10/10 | 0/10 |
| tree_hierarchy | cpu | low | 10/10 | 0/10 |
| tree_hierarchy | cpu | medium | 10/10 | 0/10 |
| tree_hierarchy | cpu | high | 10/10 | 0/10 |
| tree_hierarchy | cuda | low | 10/10 | 0/10 |
| tree_hierarchy | cuda | medium | 10/10 | 0/10 |
| tree_hierarchy | cuda | high | 10/10 | 0/10 |

## Determinism components tested

1. **Canonical traversal**: Z-order (Morton) tie-break when rho values are equal
2. **Deterministic probe**: SHA-256 hash of (coords, level, global_seed)
3. **Governor isolation**: EMA update only after full step commit
