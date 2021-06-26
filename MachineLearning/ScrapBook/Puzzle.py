#%%
import numpy as np
#%%
d = { "01" : "A","1000": "B","1010": "C","100": "D","0": "E","0010": "F","110": "G","0000": "H","00": "I","0111": "J","101": "K","0100": "L","11":"M","10": "N","111": "O","0110": "P","1101":"Q","010": "R","000": "S","1":"T","001":"U","0001":"V","011":"W","1001":"X","1011":"Y","1100":"Z",        }
d
# %%
d['001']
# %%
kl = np.array([len(key) for key,value in d.items()])
# %%
for key, value in d.items():
    print(key, value)
# %%
kl.max()
#%%
class WordPositions:
    def __init__(self, position, word):
        self.position = position
        self.word = word
    def __str__(self) -> str:
        return f'p - {self.position}, v - {self.word}'

# %%
# 0011
# 1- 0,0,1,1
# 2 - 00,01,11
# 3 - 001,011
# 4 - 0011
def rec(s, head, resultlist):
    if head >= len(s):
        return
    for i in np.arange(1,5):
        # branch out every possibility in recursive manner 
        if(head+i < (len(s)+1) and  s[head:head + i] in d):
            resultlist.append(WordPositions(head,d[s[head:head + i]]))
            rec(s,head+i, resultlist)
           # print(result)
    return
#%%
def getPrev(position, resultlist, anchor) -> str:
    result=''
    for wp in resultlist[anchor: len(resultlist)]:
        if(wp.position<position):
            result+=wp.word
        else:
            break
    return result


#%%  
s="110111010100"
resultlist = []
rec(s,0, resultlist)
for item in resultlist:
    print(item)
#%%
head=0
i=0
prev = -1
wlist=''
for wp in resultlist:
    if(wp.position==0):
        head=i
    if(wp.position>prev):
        wlist+=wp.word
        prev=wp.position
    else:
        wlist+=' ' + getPrev(wp.position,resultlist,head) + wp.word
        prev = wp.position
    i+=1
wlist
# %%
