# TopoGuard hypothesis tests (pre-registered)

## H1: overall method ranking (mean final_acc)

| method | mean_acc | n |
| --- | --- | --- |
| geomedian | 0.9656 | 270 |
| tg_trustonly | 0.9626 | 270 |
| balance_fixed | 0.9375 | 270 |
| krum | 0.9368 | 270 |
| topoguard | 0.9214 | 270 |
| trimmedmean | 0.8403 | 270 |
| fedavg | 0.7002 | 270 |

## H2/H3: per-topology ranking and cross-topology variance

| topology | geomedian | tg_trustonly | balance_fixed | krum | topoguard | trimmedmean | fedavg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ring | 0.9501 | 0.9487 | 0.9447 | 0.9013 | 0.9478 | 0.7213 | 0.7213 |
| star | 0.9703 | 0.9647 | 0.8892 | 0.9504 | 0.9076 | 0.9065 | 0.7488 |
| complete | 0.9764 | 0.9743 | 0.9786 | 0.9587 | 0.9087 | 0.8932 | 0.6307 |

Cross-topology std per method (lower = more topology-robust):
| method | std_across_topos |
| --- | --- |
| geomedian | 0.0112 |
| tg_trustonly | 0.0106 |
| balance_fixed | 0.0368 |
| krum | 0.0253 |
| topoguard | 0.0187 |
| trimmedmean | 0.0844 |
| fedavg | 0.0504 |

## H7: selection value (topoguard minus tg_trustonly)

| topology | attack | delta | n |
| --- | --- | --- | --- |
| ring | label_flip | -0.0003 | 18 |
| ring | sign_flip | +0.0022 | 18 |
| ring | gaussian | -0.0060 | 18 |
| ring | backdoor | -0.0002 | 18 |
| ring | alie | -0.0003 | 18 |
| star | label_flip | -0.0035 | 18 |
| star | sign_flip | -0.2211 | 18 |
| star | gaussian | -0.0359 | 18 |
| star | backdoor | +0.0031 | 18 |
| star | alie | -0.0282 | 18 |
| complete | label_flip | -0.0381 | 18 |
| complete | sign_flip | -0.2186 | 18 |
| complete | gaussian | -0.0220 | 18 |
| complete | backdoor | -0.0002 | 18 |
| complete | alie | -0.0489 | 18 |

## H4: backdoor ASR by defense (lower is better)

| defense | mean_ASR | mean_acc | n |
| --- | --- | --- | --- |
| geomedian | 0.5513 | 0.9728 | 54 |
| tg_trustonly | 0.5543 | 0.9677 | 54 |
| balance_fixed | 0.5405 | 0.9423 | 54 |
| krum | 0.4508 | 0.9391 | 54 |
| topoguard | 0.5831 | 0.9686 | 54 |
| trimmedmean | 0.6430 | 0.9766 | 54 |
| fedavg | 0.8198 | 0.9756 | 54 |

## Byzantine fraction sensitivity (mean_acc)

| defense | 10% | 20% | 30% |
| --- | --- | --- | --- |
| geomedian | 0.9699 | 0.9651 | 0.9618 |
| tg_trustonly | 0.9657 | 0.9604 | 0.9616 |
| balance_fixed | 0.9376 | 0.9395 | 0.9354 |
| krum | 0.9455 | 0.9199 | 0.9450 |
| topoguard | 0.9694 | 0.9640 | 0.8307 |
| trimmedmean | 0.9112 | 0.8879 | 0.7219 |
| fedavg | 0.7733 | 0.6814 | 0.6460 |
