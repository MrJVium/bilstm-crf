import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def count(file_path):

    lens = {}
    max_len = 0
    max_str = ''
    index = 0
    for line in open(file_path):
        s = ''.join(line.strip().split())
        if s == '':
            continue
        index += 1
        # if index < 5:
        #     print(s)

        if len(s) in lens:
            lens[len(s)] += 1
        else:
            lens[len(s)] = 1

        if max_len < len(s):
            max_len = len(s)
            max_str = s
    xs = []
    ys = []
    for i in sorted(lens):
        xs.append(i)
        ys.append(lens[i])
    return max_str, max_len, index, xs, ys

msr = count("seg-data/training/msr_training.utf8")
pku = count("seg-data/training/pku_training.utf8")


# plt.bar(msr[3], msr[4], label='msr')
print(pku[:-2])
plt.bar(pku[3], pku[4], label='pku')

plt.legend()

plt.xlabel('number')
plt.ylabel('value')
plt.show()