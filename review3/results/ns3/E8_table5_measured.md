Attackers blocked, mean over seeds (n).

| Measured cost (source) | Rule | 20 bots | 40 bots | 60 bots | 80 bots |
|---|---|---:|---:|---:|---:|
| 456.7 µs (x86-idle-p50) | A flat (base paper) | 100% (20) | 100% (20) | 100% (20) | 100% (20) |
| 456.7 µs (x86-idle-p50) | B EWMA | 100% (20) | 100% (20) | 100% (20) | 100% (20) |
| 456.7 µs (x86-idle-p50) | C criticality | 100% (20) | 100% (20) | 100% (10) | 100% (10) |
| 4492.7 µs (x86-load-p95) | A flat (base paper) | 100% (10) | 100% (10) | 100% (10) | 100% (10) |
| 4492.7 µs (x86-load-p95) | B EWMA | 100% (10) | 0% (10) | 0% (10) | 0% (10) |
| 4492.7 µs (x86-load-p95) | B' hysteresis | 100% (10) | 100% (10) | 100% (10) | 100% (10) |
| 4492.7 µs (x86-load-p95) | C criticality | 100% (10) | 100% (10) | 100% (10) | 100% (10) |
