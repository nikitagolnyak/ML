Nmk = str(input())
Nmk = Nmk.split(' ')
N = int(Nmk[0])
m = int(Nmk[1])
k = int(Nmk[2])

arr = list(map(int, input().split()))
elements = []
for i in range(m):
    elements.append([])
counter = 1
for i in arr:
    elements[i - 1].append(counter)
    counter = counter + 1
start = j = i = 0
temp = []
new = []
for el in elements:
    for e in el:
        new.append(e)
while i < N:
    j = start
    while j < N:
        temp.append(str(new[j]))
        j = j + k
        i = i + 1
    print(len(temp), " ".join(temp))
    temp.clear()
    start = start + 1