# TopoGuard results (measured)

## Pilot: mechanism selection (final_acc, mean of 2 seeds)
| variant/topology/etc | complete/backdoor | complete/sign_flip | ring/backdoor | ring/sign_flip |
| --- | --- | --- | --- | --- |
| V_fused | 0.9835 | 0.985 | 0.9848 | 0.9141 |
| V_select | 0.9835 | 0.985 | 0.9847 | 0.9847 |
| V_trust | 0.9849 | 0.9865 | 0.9849 | 0.9847 |


**Mean over pilot cells:** V_fused=0.9669, V_select=0.9845, V_trust=0.9853

## Tune: hyperparameter grid (mean final_acc over ring/star x sign_flip/backdoor, non-IID)
| hyperparams | mean_acc | n |
| --- | --- | --- |
| c=0.25 kappa=0.5 tau3=0.7 | 0.961 | 4 |
| c=0.25 kappa=0.5 tau3=0.9 | 0.961 | 4 |
| c=0.25 kappa=1.0 tau3=0.7 | 0.9609 | 4 |
| c=0.25 kappa=1.0 tau3=0.9 | 0.9609 | 4 |
| c=0.25 kappa=2.0 tau3=0.7 | 0.9608 | 4 |
| c=0.25 kappa=2.0 tau3=0.9 | 0.9608 | 4 |
| c=0.5 kappa=2.0 tau3=0.9 | 0.9592 | 4 |
| c=0.75 kappa=0.5 tau3=0.7 | 0.9592 | 4 |
| c=0.5 kappa=1.0 tau3=0.7 | 0.9591 | 4 |
| c=0.75 kappa=1.0 tau3=0.7 | 0.9591 | 4 |
| c=0.5 kappa=2.0 tau3=0.7 | 0.959 | 4 |
| c=0.5 kappa=1.0 tau3=0.9 | 0.9587 | 4 |
| c=0.75 kappa=2.0 tau3=0.7 | 0.9587 | 4 |
| c=0.5 kappa=0.5 tau3=0.7 | 0.958 | 4 |
| c=0.5 kappa=0.5 tau3=0.9 | 0.9577 | 4 |
| c=0.75 kappa=1.0 tau3=0.9 | 0.9569 | 4 |
| c=0.75 kappa=2.0 tau3=0.9 | 0.9566 | 4 |
| c=0.75 kappa=0.5 tau3=0.9 | 0.9547 | 4 |


## Main: complete — defense x attack (final_acc, mean over frac/data/seeds)
| variant/topology/etc | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| balance_fixed | 0.9785 | 0.9788 | 0.9787 | 0.9782 | 0.9786 |
| fedavg | 0.7851 | 0.9746 | 0.7461 | 0.5419 | 0.1058 |
| geomedian | 0.9693 | 0.9811 | 0.9809 | 0.9705 | 0.9799 |
| krum | 0.9614 | 0.9581 | 0.9583 | 0.9573 | 0.9583 |
| tg_trustonly | 0.9814 | 0.9814 | 0.9481 | 0.9807 | 0.9799 |
| topoguard | 0.9325 | 0.9813 | 0.9261 | 0.9426 | 0.7613 |
| trimmedmean | 0.8997 | 0.981 | 0.9289 | 0.9375 | 0.7191 |


### complete backdoor ASR (lower is better)
| variant/topology/etc | frac=0.1 | frac=0.2 | frac=0.3 |
| --- | --- | --- | --- |
| balance_fixed | 0.5238 | 0.5402 | 0.6283 |
| fedavg | 0.8873 | 0.9743 | 0.9843 |
| geomedian | 0.5324 | 0.5871 | 0.6369 |
| krum | 0.5239 | 0.4854 | 0.4623 |
| tg_trustonly | 0.5246 | 0.5685 | 0.6913 |
| topoguard | 0.5209 | 0.5967 | 0.8417 |
| trimmedmean | 0.5137 | 0.5442 | 0.7853 |


## Main: ring — defense x attack (final_acc, mean over frac/data/seeds)
| variant/topology/etc | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| balance_fixed | 0.9264 | 0.9574 | 0.9632 | 0.9132 | 0.9633 |
| fedavg | 0.8612 | 0.9728 | 0.7582 | 0.7428 | 0.2714 |
| geomedian | 0.9412 | 0.9612 | 0.9568 | 0.9403 | 0.9512 |
| krum | 0.8999 | 0.9123 | 0.8973 | 0.9 | 0.8973 |
| tg_trustonly | 0.9482 | 0.9486 | 0.9589 | 0.9368 | 0.951 |
| topoguard | 0.948 | 0.9484 | 0.9529 | 0.9365 | 0.9532 |
| trimmedmean | 0.8612 | 0.9728 | 0.7582 | 0.7428 | 0.2714 |


### ring backdoor ASR (lower is better)
| variant/topology/etc | frac=0.1 | frac=0.2 | frac=0.3 |
| --- | --- | --- | --- |
| balance_fixed | 0.5108 | 0.545 | 0.593 |
| fedavg | 0.6912 | 0.776 | 0.8523 |
| geomedian | 0.5043 | 0.5214 | 0.5689 |
| krum | 0.4285 | 0.4232 | 0.4575 |
| tg_trustonly | 0.5015 | 0.5012 | 0.5732 |
| topoguard | 0.4992 | 0.4998 | 0.573 |
| trimmedmean | 0.6912 | 0.776 | 0.8523 |


## Main: star — defense x attack (final_acc, mean over frac/data/seeds)
| variant/topology/etc | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| balance_fixed | 0.8871 | 0.8907 | 0.8909 | 0.8863 | 0.8909 |
| fedavg | 0.8772 | 0.9794 | 0.8031 | 0.9323 | 0.1518 |
| geomedian | 0.9634 | 0.9761 | 0.9745 | 0.9643 | 0.9731 |
| krum | 0.9552 | 0.9469 | 0.9527 | 0.9447 | 0.9527 |
| tg_trustonly | 0.9461 | 0.973 | 0.9615 | 0.9693 | 0.9735 |
| topoguard | 0.9179 | 0.9761 | 0.9256 | 0.9658 | 0.7524 |
| trimmedmean | 0.9119 | 0.9761 | 0.927 | 0.9653 | 0.7519 |


### star backdoor ASR (lower is better)
| variant/topology/etc | frac=0.1 | frac=0.2 | frac=0.3 |
| --- | --- | --- | --- |
| balance_fixed | 0.5072 | 0.5111 | 0.5053 |
| fedavg | 0.6244 | 0.7609 | 0.8278 |
| geomedian | 0.5042 | 0.5419 | 0.565 |
| krum | 0.4341 | 0.3975 | 0.4444 |
| tg_trustonly | 0.4751 | 0.5288 | 0.6245 |
| topoguard | 0.4845 | 0.509 | 0.7226 |
| trimmedmean | 0.4865 | 0.5016 | 0.636 |


## Held-out family: random_geometric (never tuned on)
| variant/topology/etc | alie | backdoor | gaussian | label_flip | sign_flip |
| --- | --- | --- | --- | --- | --- |
| balance_fixed | 0.9557 | 0.9697 | 0.9697 | 0.9376 | 0.9698 |
| fedavg | 0.7968 | 0.9676 | 0.7254 | 0.6627 | 0.135 |
| topoguard | 0.9222 | 0.9683 | 0.9423 | 0.889 | 0.8837 |


## Ablations (final_acc, non-IID, 20% Byzantine)
| variant/topology/etc | complete/backdoor | complete/sign_flip | ring/backdoor | ring/sign_flip | star/backdoor | star/sign_flip |
| --- | --- | --- | --- | --- | --- | --- |
| A1_no_select | 0.9804 | 0.9812 | 0.9569 | 0.959 | 0.9706 | 0.9733 |
| A2_no_topology | 0.9822 | 0.9715 | 0.9564 | 0.9582 | 0.9764 | 0.9674 |
| A3_no_trust | 0.977 | 0.2657 | 0.9799 | 0.41 | 0.9802 | 0.2688 |
| A4_small_portfolio | 0.9828 | 0.9759 | 0.9657 | 0.9354 | 0.971 | 0.9625 |
| full | 0.9821 | 0.9748 | 0.9569 | 0.959 | 0.9755 | 0.9715 |


## Time-varying topology stress
| variant/topology/etc | backdoor | sign_flip |
| --- | --- | --- |
| balance_fixed | 0.9839 | 0.984 |
| fedavg | 0.9818 | 0.3723 |
| topoguard | 0.9843 | 0.9844 |
