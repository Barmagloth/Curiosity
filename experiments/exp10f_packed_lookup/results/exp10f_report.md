# Exp10f: Packed Tile Storage with Direct Lookup -- Report

## Kill Criteria: >20% overhead vs grid in wall-clock OR VRAM

### A_bitset: ALIVE
  - random: PASS  (time -19.9%, VRAM +17.8%)
  - clustered: PASS  (time -20.6%, VRAM +17.8%)
  - checkerboard: PASS  (time -20.6%, VRAM +17.8%)

### D_direct_onfly: KILLED
  - random: FAIL  (VRAM +230.4%)
  - clustered: FAIL  (VRAM +130.4%)
  - checkerboard: FAIL  (VRAM +230.4%)

### D_direct_prebuilt: KILLED
  - random: FAIL  (VRAM +196.0%)
  - clustered: FAIL  (VRAM +106.9%)
  - checkerboard: FAIL  (VRAM +196.0%)

### E_hash_onfly: KILLED
  - random: FAIL  (VRAM +228.2%)
  - clustered: FAIL  (VRAM +129.1%)
  - checkerboard: FAIL  (VRAM +228.2%)

### E_hash_prebuilt: KILLED
  - random: FAIL  (VRAM +196.9%)
  - clustered: FAIL  (VRAM +107.8%)
  - checkerboard: FAIL  (VRAM +196.9%)

## Competitive Analysis vs A_bitset

### D_direct_onfly
  - random: time vs A = -77.2%, VRAM vs A = +180.3%, sparse VRAM vs A = +208.6%
  - clustered: time vs A = -78.4%, VRAM vs A = +107.1%, sparse VRAM vs A = +55.8%
  - checkerboard: time vs A = -77.5%, VRAM vs A = +180.3%, sparse VRAM vs A = +208.6%

### D_direct_prebuilt
  - random: time vs A = -77.5%, VRAM vs A = +151.2%, sparse VRAM vs A = +175.5%
  - clustered: time vs A = -77.6%, VRAM vs A = +85.7%, sparse VRAM vs A = +41.2%
  - checkerboard: time vs A = -77.6%, VRAM vs A = +151.2%, sparse VRAM vs A = +175.5%

### E_hash_onfly
  - random: time vs A = -77.3%, VRAM vs A = +178.5%, sparse VRAM vs A = +206.6%
  - clustered: time vs A = -78.5%, VRAM vs A = +105.8%, sparse VRAM vs A = +54.7%
  - checkerboard: time vs A = -77.8%, VRAM vs A = +178.5%, sparse VRAM vs A = +206.6%

### E_hash_prebuilt
  - random: time vs A = -77.6%, VRAM vs A = +152.0%, sparse VRAM vs A = +176.2%
  - clustered: time vs A = -79.0%, VRAM vs A = +86.4%, sparse VRAM vs A = +41.5%
  - checkerboard: time vs A = -77.9%, VRAM vs A = +152.0%, sparse VRAM vs A = +176.2%

## Build Cost Analysis

### D_direct_onfly -- build overhead warnings:
  - side=64_sp=0.05_pat=random: build=0.42ms (522% of compute)
  - side=64_sp=0.05_pat=clustered: build=0.43ms (551% of compute)
  - side=64_sp=0.05_pat=checkerboard: build=0.40ms (518% of compute)
  - side=64_sp=0.1_pat=random: build=0.41ms (531% of compute)
  - side=64_sp=0.1_pat=clustered: build=0.44ms (558% of compute)
  - side=64_sp=0.1_pat=checkerboard: build=0.44ms (569% of compute)
  - side=64_sp=0.2_pat=random: build=0.41ms (528% of compute)
  - side=64_sp=0.2_pat=clustered: build=0.42ms (537% of compute)
  - side=64_sp=0.2_pat=checkerboard: build=0.42ms (536% of compute)
  - side=64_sp=0.3_pat=random: build=0.42ms (532% of compute)
  - side=64_sp=0.3_pat=clustered: build=0.41ms (529% of compute)
  - side=64_sp=0.3_pat=checkerboard: build=0.50ms (642% of compute)
  - side=64_sp=0.5_pat=random: build=0.39ms (527% of compute)
  - side=64_sp=0.5_pat=clustered: build=0.44ms (638% of compute)
  - side=64_sp=0.5_pat=checkerboard: build=0.42ms (597% of compute)
  - side=64_sp=0.7_pat=random: build=0.41ms (591% of compute)
  - side=64_sp=0.7_pat=clustered: build=0.42ms (616% of compute)
  - side=64_sp=0.7_pat=checkerboard: build=0.45ms (646% of compute)
  - side=128_sp=0.05_pat=random: build=0.40ms (555% of compute)
  - side=128_sp=0.05_pat=clustered: build=0.41ms (607% of compute)
  - side=128_sp=0.05_pat=checkerboard: build=0.41ms (553% of compute)
  - side=128_sp=0.1_pat=random: build=0.41ms (598% of compute)
  - side=128_sp=0.1_pat=clustered: build=0.43ms (588% of compute)
  - side=128_sp=0.1_pat=checkerboard: build=0.43ms (557% of compute)
  - side=128_sp=0.2_pat=random: build=0.42ms (622% of compute)
  - side=128_sp=0.2_pat=clustered: build=0.44ms (675% of compute)
  - side=128_sp=0.2_pat=checkerboard: build=0.42ms (590% of compute)
  - side=128_sp=0.3_pat=random: build=0.42ms (600% of compute)
  - side=128_sp=0.3_pat=clustered: build=0.53ms (693% of compute)
  - side=128_sp=0.3_pat=checkerboard: build=0.44ms (669% of compute)
  - side=128_sp=0.5_pat=random: build=0.42ms (585% of compute)
  - side=128_sp=0.5_pat=clustered: build=0.48ms (684% of compute)
  - side=128_sp=0.5_pat=checkerboard: build=0.45ms (645% of compute)
  - side=128_sp=0.7_pat=random: build=0.40ms (599% of compute)
  - side=128_sp=0.7_pat=clustered: build=0.47ms (637% of compute)
  - side=128_sp=0.7_pat=checkerboard: build=0.44ms (615% of compute)
  - side=256_sp=0.05_pat=random: build=0.46ms (407% of compute)
  - side=256_sp=0.05_pat=clustered: build=0.48ms (712% of compute)
  - side=256_sp=0.05_pat=checkerboard: build=0.46ms (408% of compute)
  - side=256_sp=0.1_pat=random: build=0.48ms (401% of compute)
  - side=256_sp=0.1_pat=clustered: build=0.48ms (636% of compute)
  - side=256_sp=0.1_pat=checkerboard: build=0.44ms (454% of compute)
  - side=256_sp=0.2_pat=random: build=0.44ms (455% of compute)
  - side=256_sp=0.2_pat=clustered: build=0.48ms (622% of compute)
  - side=256_sp=0.2_pat=checkerboard: build=0.44ms (568% of compute)
  - side=256_sp=0.3_pat=random: build=0.41ms (536% of compute)
  - side=256_sp=0.3_pat=clustered: build=0.55ms (558% of compute)
  - side=256_sp=0.3_pat=checkerboard: build=0.48ms (377% of compute)
  - side=256_sp=0.5_pat=random: build=0.48ms (385% of compute)
  - side=256_sp=0.5_pat=clustered: build=0.52ms (417% of compute)
  - side=256_sp=0.5_pat=checkerboard: build=0.49ms (537% of compute)
  - side=256_sp=0.7_pat=random: build=0.48ms (504% of compute)
  - side=256_sp=0.7_pat=clustered: build=0.50ms (371% of compute)
  - side=256_sp=0.7_pat=checkerboard: build=0.43ms (495% of compute)

### D_direct_prebuilt -- build overhead warnings:
  - side=64_sp=0.05_pat=random: build=0.40ms (508% of compute)
  - side=64_sp=0.05_pat=clustered: build=0.43ms (550% of compute)
  - side=64_sp=0.05_pat=checkerboard: build=0.39ms (524% of compute)
  - side=64_sp=0.1_pat=random: build=0.41ms (542% of compute)
  - side=64_sp=0.1_pat=clustered: build=0.44ms (572% of compute)
  - side=64_sp=0.1_pat=checkerboard: build=0.42ms (549% of compute)
  - side=64_sp=0.2_pat=random: build=0.40ms (515% of compute)
  - side=64_sp=0.2_pat=clustered: build=0.41ms (536% of compute)
  - side=64_sp=0.2_pat=checkerboard: build=0.43ms (547% of compute)
  - side=64_sp=0.3_pat=random: build=0.41ms (529% of compute)
  - side=64_sp=0.3_pat=clustered: build=0.42ms (537% of compute)
  - side=64_sp=0.3_pat=checkerboard: build=0.41ms (528% of compute)
  - side=64_sp=0.5_pat=random: build=0.43ms (635% of compute)
  - side=64_sp=0.5_pat=clustered: build=0.47ms (668% of compute)
  - side=64_sp=0.5_pat=checkerboard: build=0.41ms (602% of compute)
  - side=64_sp=0.7_pat=random: build=0.42ms (602% of compute)
  - side=64_sp=0.7_pat=clustered: build=0.42ms (616% of compute)
  - side=64_sp=0.7_pat=checkerboard: build=0.45ms (640% of compute)
  - side=128_sp=0.05_pat=random: build=0.44ms (597% of compute)
  - side=128_sp=0.05_pat=clustered: build=0.44ms (644% of compute)
  - side=128_sp=0.05_pat=checkerboard: build=0.40ms (547% of compute)
  - side=128_sp=0.1_pat=random: build=0.38ms (554% of compute)
  - side=128_sp=0.1_pat=clustered: build=0.49ms (630% of compute)
  - side=128_sp=0.1_pat=checkerboard: build=0.43ms (583% of compute)
  - side=128_sp=0.2_pat=random: build=0.40ms (584% of compute)
  - side=128_sp=0.2_pat=clustered: build=0.45ms (694% of compute)
  - side=128_sp=0.2_pat=checkerboard: build=0.42ms (594% of compute)
  - side=128_sp=0.3_pat=random: build=0.45ms (641% of compute)
  - side=128_sp=0.3_pat=clustered: build=0.55ms (719% of compute)
  - side=128_sp=0.3_pat=checkerboard: build=0.41ms (630% of compute)
  - side=128_sp=0.5_pat=random: build=0.44ms (630% of compute)
  - side=128_sp=0.5_pat=clustered: build=0.46ms (646% of compute)
  - side=128_sp=0.5_pat=checkerboard: build=0.42ms (604% of compute)
  - side=128_sp=0.7_pat=random: build=0.39ms (577% of compute)
  - side=128_sp=0.7_pat=clustered: build=0.44ms (612% of compute)
  - side=128_sp=0.7_pat=checkerboard: build=0.43ms (618% of compute)
  - side=256_sp=0.05_pat=random: build=0.46ms (403% of compute)
  - side=256_sp=0.05_pat=clustered: build=0.48ms (706% of compute)
  - side=256_sp=0.05_pat=checkerboard: build=0.46ms (411% of compute)
  - side=256_sp=0.1_pat=random: build=0.48ms (408% of compute)
  - side=256_sp=0.1_pat=clustered: build=0.49ms (629% of compute)
  - side=256_sp=0.1_pat=checkerboard: build=0.42ms (438% of compute)
  - side=256_sp=0.2_pat=random: build=0.43ms (442% of compute)
  - side=256_sp=0.2_pat=clustered: build=0.53ms (627% of compute)
  - side=256_sp=0.2_pat=checkerboard: build=0.45ms (576% of compute)
  - side=256_sp=0.3_pat=random: build=0.42ms (551% of compute)
  - side=256_sp=0.3_pat=clustered: build=0.51ms (468% of compute)
  - side=256_sp=0.3_pat=checkerboard: build=0.47ms (373% of compute)
  - side=256_sp=0.5_pat=random: build=0.46ms (374% of compute)
  - side=256_sp=0.5_pat=clustered: build=0.54ms (424% of compute)
  - side=256_sp=0.5_pat=checkerboard: build=0.42ms (476% of compute)
  - side=256_sp=0.7_pat=random: build=0.63ms (529% of compute)
  - side=256_sp=0.7_pat=clustered: build=0.57ms (233% of compute)
  - side=256_sp=0.7_pat=checkerboard: build=0.49ms (563% of compute)

### E_hash_onfly -- build overhead warnings:
  - side=64_sp=0.05_pat=random: build=2.23ms (2850% of compute)
  - side=64_sp=0.05_pat=clustered: build=4.50ms (5806% of compute)
  - side=64_sp=0.05_pat=checkerboard: build=2.44ms (3230% of compute)
  - side=64_sp=0.1_pat=random: build=2.53ms (3279% of compute)
  - side=64_sp=0.1_pat=clustered: build=2.71ms (3462% of compute)
  - side=64_sp=0.1_pat=checkerboard: build=2.65ms (3395% of compute)
  - side=64_sp=0.2_pat=random: build=2.30ms (3006% of compute)
  - side=64_sp=0.2_pat=clustered: build=2.48ms (3820% of compute)
  - side=64_sp=0.2_pat=checkerboard: build=2.30ms (2946% of compute)
  - side=64_sp=0.3_pat=random: build=3.64ms (4659% of compute)
  - side=64_sp=0.3_pat=clustered: build=2.61ms (3366% of compute)
  - side=64_sp=0.3_pat=checkerboard: build=2.51ms (3272% of compute)
  - side=64_sp=0.5_pat=random: build=2.51ms (3714% of compute)
  - side=64_sp=0.5_pat=clustered: build=2.68ms (3854% of compute)
  - side=64_sp=0.5_pat=checkerboard: build=2.61ms (3757% of compute)
  - side=64_sp=0.7_pat=random: build=2.45ms (3186% of compute)
  - side=64_sp=0.7_pat=clustered: build=2.66ms (3824% of compute)
  - side=64_sp=0.7_pat=checkerboard: build=2.52ms (3303% of compute)
  - side=128_sp=0.05_pat=random: build=4.41ms (6138% of compute)
  - side=128_sp=0.05_pat=clustered: build=6.35ms (9126% of compute)
  - side=128_sp=0.05_pat=checkerboard: build=4.36ms (5918% of compute)
  - side=128_sp=0.1_pat=random: build=4.30ms (6348% of compute)
  - side=128_sp=0.1_pat=clustered: build=6.86ms (8823% of compute)
  - side=128_sp=0.1_pat=checkerboard: build=4.51ms (6908% of compute)
  - side=128_sp=0.2_pat=random: build=4.55ms (6617% of compute)
  - side=128_sp=0.2_pat=clustered: build=2.57ms (3974% of compute)
  - side=128_sp=0.2_pat=checkerboard: build=4.65ms (6604% of compute)
  - side=128_sp=0.3_pat=random: build=5.08ms (6469% of compute)
  - side=128_sp=0.3_pat=clustered: build=2.66ms (3948% of compute)
  - side=128_sp=0.3_pat=checkerboard: build=4.43ms (6683% of compute)
  - side=128_sp=0.5_pat=random: build=4.37ms (6268% of compute)
  - side=128_sp=0.5_pat=clustered: build=4.74ms (6779% of compute)
  - side=128_sp=0.5_pat=checkerboard: build=4.56ms (6865% of compute)
  - side=128_sp=0.7_pat=random: build=4.55ms (6517% of compute)
  - side=128_sp=0.7_pat=clustered: build=4.69ms (6512% of compute)
  - side=128_sp=0.7_pat=checkerboard: build=4.47ms (6375% of compute)
  - side=256_sp=0.05_pat=random: build=5.63ms (5731% of compute)
  - side=256_sp=0.05_pat=clustered: build=14.56ms (21469% of compute)
  - side=256_sp=0.05_pat=checkerboard: build=4.70ms (4121% of compute)
  - side=256_sp=0.1_pat=random: build=4.59ms (4257% of compute)
  - side=256_sp=0.1_pat=clustered: build=6.61ms (8505% of compute)
  - side=256_sp=0.1_pat=checkerboard: build=4.58ms (4765% of compute)
  - side=256_sp=0.2_pat=random: build=4.49ms (4677% of compute)
  - side=256_sp=0.2_pat=clustered: build=10.27ms (13209% of compute)
  - side=256_sp=0.2_pat=checkerboard: build=4.75ms (6099% of compute)
  - side=256_sp=0.3_pat=random: build=4.76ms (6256% of compute)
  - side=256_sp=0.3_pat=clustered: build=4.91ms (4525% of compute)
  - side=256_sp=0.3_pat=checkerboard: build=4.67ms (3741% of compute)
  - side=256_sp=0.5_pat=random: build=4.67ms (3865% of compute)
  - side=256_sp=0.5_pat=clustered: build=5.02ms (4878% of compute)
  - side=256_sp=0.5_pat=checkerboard: build=4.36ms (4988% of compute)
  - side=256_sp=0.7_pat=random: build=5.87ms (6842% of compute)
  - side=256_sp=0.7_pat=clustered: build=4.96ms (3729% of compute)
  - side=256_sp=0.7_pat=checkerboard: build=4.10ms (5585% of compute)

### E_hash_prebuilt -- build overhead warnings:
  - side=64_sp=0.05_pat=random: build=2.21ms (2896% of compute)
  - side=64_sp=0.05_pat=clustered: build=5.14ms (6667% of compute)
  - side=64_sp=0.05_pat=checkerboard: build=2.96ms (3798% of compute)
  - side=64_sp=0.1_pat=random: build=2.89ms (3692% of compute)
  - side=64_sp=0.1_pat=clustered: build=3.39ms (4244% of compute)
  - side=64_sp=0.1_pat=checkerboard: build=2.33ms (2931% of compute)
  - side=64_sp=0.2_pat=random: build=2.41ms (3128% of compute)
  - side=64_sp=0.2_pat=clustered: build=2.51ms (3830% of compute)
  - side=64_sp=0.2_pat=checkerboard: build=2.36ms (3019% of compute)
  - side=64_sp=0.3_pat=random: build=2.44ms (3128% of compute)
  - side=64_sp=0.3_pat=clustered: build=2.67ms (3432% of compute)
  - side=64_sp=0.3_pat=checkerboard: build=2.44ms (3222% of compute)
  - side=64_sp=0.5_pat=random: build=2.49ms (3678% of compute)
  - side=64_sp=0.5_pat=clustered: build=2.69ms (3850% of compute)
  - side=64_sp=0.5_pat=checkerboard: build=2.62ms (3774% of compute)
  - side=64_sp=0.7_pat=random: build=2.53ms (3257% of compute)
  - side=64_sp=0.7_pat=clustered: build=2.83ms (4028% of compute)
  - side=64_sp=0.7_pat=checkerboard: build=3.27ms (4215% of compute)
  - side=128_sp=0.05_pat=random: build=4.31ms (6004% of compute)
  - side=128_sp=0.05_pat=clustered: build=6.03ms (8988% of compute)
  - side=128_sp=0.05_pat=checkerboard: build=6.73ms (9257% of compute)
  - side=128_sp=0.1_pat=random: build=4.42ms (6748% of compute)
  - side=128_sp=0.1_pat=clustered: build=7.38ms (9489% of compute)
  - side=128_sp=0.1_pat=checkerboard: build=4.58ms (6996% of compute)
  - side=128_sp=0.2_pat=random: build=4.53ms (6507% of compute)
  - side=128_sp=0.2_pat=clustered: build=2.90ms (4501% of compute)
  - side=128_sp=0.2_pat=checkerboard: build=4.53ms (6307% of compute)
  - side=128_sp=0.3_pat=random: build=4.10ms (5286% of compute)
  - side=128_sp=0.3_pat=clustered: build=2.73ms (4027% of compute)
  - side=128_sp=0.3_pat=checkerboard: build=4.87ms (7160% of compute)
  - side=128_sp=0.5_pat=random: build=4.39ms (6491% of compute)
  - side=128_sp=0.5_pat=clustered: build=4.97ms (7373% of compute)
  - side=128_sp=0.5_pat=checkerboard: build=4.45ms (6797% of compute)
  - side=128_sp=0.7_pat=random: build=4.59ms (6426% of compute)
  - side=128_sp=0.7_pat=clustered: build=5.40ms (7532% of compute)
  - side=128_sp=0.7_pat=checkerboard: build=4.43ms (6325% of compute)
  - side=256_sp=0.05_pat=random: build=4.85ms (4985% of compute)
  - side=256_sp=0.05_pat=clustered: build=14.18ms (20077% of compute)
  - side=256_sp=0.05_pat=checkerboard: build=4.69ms (4128% of compute)
  - side=256_sp=0.1_pat=random: build=4.41ms (4454% of compute)
  - side=256_sp=0.1_pat=clustered: build=6.60ms (8477% of compute)
  - side=256_sp=0.1_pat=checkerboard: build=5.12ms (4749% of compute)
  - side=256_sp=0.2_pat=random: build=4.65ms (4567% of compute)
  - side=256_sp=0.2_pat=clustered: build=11.46ms (15667% of compute)
  - side=256_sp=0.2_pat=checkerboard: build=5.79ms (7463% of compute)
  - side=256_sp=0.3_pat=random: build=5.42ms (6129% of compute)
  - side=256_sp=0.3_pat=clustered: build=4.87ms (4666% of compute)
  - side=256_sp=0.3_pat=checkerboard: build=5.30ms (4333% of compute)
  - side=256_sp=0.5_pat=random: build=4.90ms (4073% of compute)
  - side=256_sp=0.5_pat=clustered: build=4.85ms (4386% of compute)
  - side=256_sp=0.5_pat=checkerboard: build=4.37ms (4981% of compute)
  - side=256_sp=0.7_pat=random: build=5.33ms (6670% of compute)
  - side=256_sp=0.7_pat=clustered: build=4.54ms (4516% of compute)
  - side=256_sp=0.7_pat=checkerboard: build=4.02ms (5505% of compute)

## Statistical Tests (Wilcoxon + Holm-Bonferroni)

### random
  - A_bitset: base=0.445ms, cand=0.347ms, p=0.0000*, oh=-21.9%
  - D_direct_onfly: base=0.445ms, cand=0.077ms, p=0.0000*, oh=-82.6%
  - D_direct_prebuilt: base=0.445ms, cand=0.076ms, p=0.0000*, oh=-82.8%
  - E_hash_onfly: base=0.445ms, cand=0.077ms, p=0.0000*, oh=-82.7%
  - E_hash_prebuilt: base=0.445ms, cand=0.078ms, p=0.0000*, oh=-82.6%

### clustered
  - A_bitset: base=0.440ms, cand=0.354ms, p=0.0000*, oh=-19.6%
  - D_direct_onfly: base=0.440ms, cand=0.076ms, p=0.0000*, oh=-82.6%
  - D_direct_prebuilt: base=0.440ms, cand=0.077ms, p=0.0000*, oh=-82.4%
  - E_hash_onfly: base=0.440ms, cand=0.075ms, p=0.0000*, oh=-83.0%
  - E_hash_prebuilt: base=0.440ms, cand=0.072ms, p=0.0000*, oh=-83.5%

### checkerboard
  - A_bitset: base=0.442ms, cand=0.349ms, p=0.0000*, oh=-21.2%
  - D_direct_onfly: base=0.442ms, cand=0.077ms, p=0.0000*, oh=-82.5%
  - D_direct_prebuilt: base=0.442ms, cand=0.076ms, p=0.0000*, oh=-82.8%
  - E_hash_onfly: base=0.442ms, cand=0.076ms, p=0.0000*, oh=-82.8%
  - E_hash_prebuilt: base=0.442ms, cand=0.077ms, p=0.0000*, oh=-82.7%
