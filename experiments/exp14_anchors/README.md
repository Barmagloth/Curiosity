## exp14 — P1-B3: Anchors + Periodic Rebuild

**Question:** Can we avoid full tree rebuilds by using local-only updates?
Two trigger strategies: periodic (every K steps) vs dirty-triggered (when >X% nodes dirty).

**Kill criteria:** Divergence < 5% vs full rebuild → local updates sufficient.

**Roadmap level:** P1-B3 (depends on exp11 dirty signatures + exp13 segment compression)

**Status:** OPEN
