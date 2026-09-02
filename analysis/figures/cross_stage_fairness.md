# Cross-stage empirical fairness

Service lead is the largest gap in cumulative, weight-normalized occupied service seconds while every tenant is backlogged at that stage. CPU and engine service are deliberately reported separately.

Engine dispatch lead counts admissions rather than concurrent residence time, because the API does not expose per-request GPU occupancy.

| Policy | Runs | Complete | Prep service lead | Prep dispatch lead | Engine dispatch lead | Worst/best tenant slowdown | Foreground E2E | Max BG prep wait | Prep util. | Engine util. | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fcfs | 4 | 100.0% | 201.0s | 12.0 | 3.5 | 1.68x | 176.1s | 142.5s | 85.2% | 43.0% | 0.210 |
| priority | 4 | 100.0% | 181.3s | 12.0 | 8.0 | 2.52x | 37.2s | 180.0s | 86.0% | 37.9% | 0.215 |
| sjf | 4 | 100.0% | 141.2s | 4.0 | 1.8 | 1.27x | 62.6s | 137.6s | 85.3% | 26.9% | 0.213 |
| tenant_fair | 4 | 100.0% | 64.2s | 6.5 | 5.5 | 1.15x | 176.0s | 170.5s | 90.7% | 45.9% | 0.226 |
| tenant_priority | 4 | 100.0% | 51.8s | 5.2 | 4.2 | 1.19x | 44.7s | 160.6s | 85.3% | 45.8% | 0.230 |
| fair_slowdown | 4 | 100.0% | 72.7s | 2.5 | 2.2 | 1.15x | 63.4s | 138.8s | 86.3% | 27.7% | 0.216 |
