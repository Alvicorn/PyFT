# PyFT benchmark results

## System info

| Key | Value |
| --- | --- |
| `python_version` | 3.14.4 (free-threaded) |
| `python_implementation` | CPython |
| `platform` | Windows-11-10.0.26200-SP0 |
| `machine` | AMD64 |
| `processor` | AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD |
| `pyft_git_sha` | 8a5da72060e9147fd6038ae42b3d57a4718cc9a7 |
| `workloads_git_sha` | b3255cfb326a63027cb4206efd54be6923619650 |
| `num_threads_setting` | 8 threads |
| `sample_interval_s` | 0.05 s |
| `cpu_count_logical` | 16 cores |
| `cpu_count_physical` | 8 cores |
| `total_ram_gb` | 27.35 GB |

## Geomean overhead (all workloads)

| Metric | v1 | v2 |
| --- | --- | --- |
| Wall-time ratio | 8.13x | 7.93x |
| Peak-memory ratio | 3.94x | 3.93x |

## Per-workload summary

| Workload | uninstr wall (s) | v1 wall (s) | v2 wall (s) | v1 wall ratio | v2 wall ratio | v1 peak resident set size ratio | v2 peak resident set size ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive_jacobi | 0.112 | 5.671 | 6.018 | 50.53x | 53.62x | 5.38x | 5.56x |
| bfs | 0.168 | 9.631 | 6.895 | 57.29x | 41.01x | 2.05x | 1.89x |
| bitonic_sort | 4.061 | 57.143 | 54.376 | 14.07x | 13.39x | 6.78x | 4.50x |
| bounded_quadsum | 1.779 | 1.995 | 1.926 | 1.12x | 1.08x | 1.84x | 1.85x |
| bounded_workers | 2.183 | 2.918 | 2.835 | 1.34x | 1.30x | 1.59x | 1.59x |
| concurrent_hashmap | 1.587 | 3.237 | 3.159 | 2.04x | 1.99x | 3.74x | 2.68x |
| cv_bounded_buffer | 1.693 | 4.588 | 6.022 | 2.71x | 3.56x | 1.73x | 6.27x |
| dining_philosophers | 1.643 | 1.853 | 1.860 | 1.13x | 1.13x | 1.85x | 1.84x |
| early_term_search | 1.598 | 2.715 | 3.499 | 1.70x | 2.19x | 1.83x | 3.00x |
| factorization_pool | 1.588 | 1.865 | 1.832 | 1.17x | 1.15x | 1.65x | 1.62x |
| fft | 2.167 | 13.689 | 11.540 | 6.32x | 5.32x | 6.96x | 3.17x |
| floyd_warshall | 1.626 | 38.890 | 38.464 | 23.92x | 23.65x | 4.98x | 4.18x |
| matmul | 1.618 | 31.641 | 32.218 | 19.56x | 19.92x | 4.51x | 3.89x |
| memo_recursion | 2.962 | 393.941 | 394.920 | 133.02x | 133.35x | 1.88x | 1.82x |
| monte_carlo_pi | 1.597 | 9.771 | 9.634 | 6.12x | 6.03x | 1.83x | 1.81x |
| nested_counter | 1.935 | 161.100 | 154.235 | 83.25x | 79.70x | 1.85x | 1.81x |
| numerical_integration | 1.719 | 30.054 | 27.886 | 17.48x | 16.22x | 1.81x | 1.82x |
| page_rank | 1.757 | 75.762 | 49.285 | 43.12x | 28.05x | 30.31x | 11.61x |
| password_crack | 1.620 | 4.044 | 4.214 | 2.50x | 2.60x | 1.87x | 1.94x |
| permit_pool | 2.907 | 197.191 | 156.527 | 67.83x | 53.84x | 468.64x | 213.60x |
| pollard_factor | 1.583 | 2.679 | 2.432 | 1.69x | 1.54x | 1.55x | 1.53x |
| prime_sieve | 1.636 | 42.726 | 37.845 | 26.12x | 23.14x | 98.01x | 50.47x |
| priority_pipeline | 1.712 | 4.873 | 6.076 | 2.85x | 3.55x | 1.87x | 6.65x |
| producer_consumer | 1.705 | 4.919 | 6.098 | 2.88x | 3.58x | 1.80x | 6.48x |
| **GEOMEAN** | - | - | - | **8.13x** | **7.93x** | **3.94x** | **3.93x** |

## Summary plots

### Wall time per benchmark (log y)

![Wall time per benchmark (log y)](plots/wall_time_per_benchmark.png)

### Peak resident set size per benchmark (log y)

![Peak resident set size per benchmark (log y)](plots/peak_memory_per_benchmark.png)

## Memory-over-time per workload

### adaptive_jacobi

![adaptive_jacobi memory over time](plots/memory_over_time/adaptive_jacobi.png)

### bfs

![bfs memory over time](plots/memory_over_time/bfs.png)

### bitonic_sort

![bitonic_sort memory over time](plots/memory_over_time/bitonic_sort.png)

### bounded_quadsum

![bounded_quadsum memory over time](plots/memory_over_time/bounded_quadsum.png)

### bounded_workers

![bounded_workers memory over time](plots/memory_over_time/bounded_workers.png)

### concurrent_hashmap

![concurrent_hashmap memory over time](plots/memory_over_time/concurrent_hashmap.png)

### cv_bounded_buffer

![cv_bounded_buffer memory over time](plots/memory_over_time/cv_bounded_buffer.png)

### dining_philosophers

![dining_philosophers memory over time](plots/memory_over_time/dining_philosophers.png)

### early_term_search

![early_term_search memory over time](plots/memory_over_time/early_term_search.png)

### factorization_pool

![factorization_pool memory over time](plots/memory_over_time/factorization_pool.png)

### fft

![fft memory over time](plots/memory_over_time/fft.png)

### floyd_warshall

![floyd_warshall memory over time](plots/memory_over_time/floyd_warshall.png)

### matmul

![matmul memory over time](plots/memory_over_time/matmul.png)

### memo_recursion

![memo_recursion memory over time](plots/memory_over_time/memo_recursion.png)

### monte_carlo_pi

![monte_carlo_pi memory over time](plots/memory_over_time/monte_carlo_pi.png)

### nested_counter

![nested_counter memory over time](plots/memory_over_time/nested_counter.png)

### numerical_integration

![numerical_integration memory over time](plots/memory_over_time/numerical_integration.png)

### page_rank

![page_rank memory over time](plots/memory_over_time/page_rank.png)

### password_crack

![password_crack memory over time](plots/memory_over_time/password_crack.png)

### permit_pool

![permit_pool memory over time](plots/memory_over_time/permit_pool.png)

### pollard_factor

![pollard_factor memory over time](plots/memory_over_time/pollard_factor.png)

### prime_sieve

![prime_sieve memory over time](plots/memory_over_time/prime_sieve.png)

### priority_pipeline

![priority_pipeline memory over time](plots/memory_over_time/priority_pipeline.png)

### producer_consumer

![producer_consumer memory over time](plots/memory_over_time/producer_consumer.png)
