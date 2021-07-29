#%%
def RunningSum(arr):
    if(len(arr)==0):
        return 0
    currentTotal = arr[0]
    maxTotal = arr[0]
    for num in arr[1:]:
        currentTotal = max(currentTotal+num, num)
        print(f'current total {currentTotal}')
        maxTotal = max(currentTotal, maxTotal)
        print(f'max total {maxTotal}' )
    return maxTotal

# %%
arr = [2,3,-10, 9, 2]
RunningSum(arr)
# %%
