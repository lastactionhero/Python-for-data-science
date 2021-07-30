#%%
# Idea is track minmum number and get max gain
def GetMaxGain(arr):
    if len(arr)==0:
        return 0
    if len(arr)==1:
        return arr[0]
    minprice = arr[0]
    gain = 0
    for price in arr[1:]:
        if price<=minprice:
            minprice=price
            continue
        if minprice < price:
            gain = max(gain,price-minprice)
    return gain
# %%
GetMaxGain([6,13,2,10,3,5])
# %%
