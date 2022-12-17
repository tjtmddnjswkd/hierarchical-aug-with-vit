import os

dt = {}
with open('/nas/datahub/tinyimagenet200/words.txt', 'r') as f:
    for i in f.readlines():
        dt[i.split()[0]] = i.split()[1]

# with open('/nas/datahub/tinyimagenet200/wnids.txt', 'r') as f:
#     bbb = f.read().split()
with open('./imagenet100.txt', 'r') as f:
    bbb = f.read().split()
words = []
for i in bbb:
    words.append(dt[i])
# for i in aaa:
#     os.system(f'cp -r /nas/datahub/imagenet/train/{i} /data2/imagenet100/train/')
import pdb;pdb.set_trace()