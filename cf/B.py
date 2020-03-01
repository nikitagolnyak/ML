n_class = int(input())
t = []
c = []
p = [0] * n_class
for i in range(n_class):
    arr = str(input()).split(" ")
    cur_sum = 0
    for j in range(n_class):
        cur = int(arr[j])
        cur_sum += cur
        if i == j:
            t.append(cur)
        if j == n_class - 1:
            c.append(cur_sum)
        p[j] += cur
precs = 0
all = sum(p)
micro = 0
for i in range(n_class):
    if p[i] == 0:
        recall = 0
    else:
        recall = t[i] / p[i]
        precs += (t[i] * c[i]) / p[i]
    if c[i] == 0:
        precision = 0
    else:
        precision = t[i] / c[i]
    if recall + precision == 0:
        fc = 0
    else:
        fc = (2.0 * recall * precision) / (recall + precision)
    micro += (c[i] * fc) / all
w_recall = sum(t) / all
w_prec = precs / all
print("Macro F1", 2.0 * (w_prec * w_recall) / (w_prec + w_recall))
print("Micro F1", micro)
