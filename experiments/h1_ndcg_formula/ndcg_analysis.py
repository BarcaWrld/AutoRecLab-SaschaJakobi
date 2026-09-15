import numpy as np


def idcg_paper(n_relevant, k):
    n_ideal_hits = min(n_relevant, k)
    return sum(1 / np.log2(i + 1) for i in range(1, n_ideal_hits + 1))


def idcg_omnirec(k):
    return sum(1 / np.log2(i + 1) for i in range(1, k + 1))


k = 10
print(f'{"n_relevant":>10} | {"IDCG_paper":>12} | {"IDCG_omnirec":>14} | {"ratio_omnirec_over_paper":>26}')
for n_relevant in [1, 2, 3, 5, 7, 9, 10, 11, 15, 20]:
    ip = idcg_paper(n_relevant, k)
    io = idcg_omnirec(k)
    ratio = io / ip
    print(f'{n_relevant:>10} | {ip:>12.4f} | {io:>14.4f} | {ratio:>26.4f}')