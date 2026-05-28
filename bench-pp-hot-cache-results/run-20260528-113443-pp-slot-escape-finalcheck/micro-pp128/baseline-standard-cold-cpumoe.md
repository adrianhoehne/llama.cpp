# baseline-standard-cold-cpumoe
| model                          |       size |     params | backend    | ngl |       cmoe | pf           | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------ | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  29.65 GiB |    34.66 B | CUDA       |  99 |          1 | pp-bench-conversation-code.txt | CUDA0        |           pp128 |         25.59 ± 0.00 |

build: 533e1177e (9402)
