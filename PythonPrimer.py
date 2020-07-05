#basic of loop
#%%
for i in [1,2,3,4]:
    print(i)
#formating example
firstName = "Jack"
lastName = "Reacher"
print(f"His is first name is {firstName}, and last name is {lastName}")
print(r"His is first name is {firstName}, and last name is {lastName}")
#%%
#List 
#%%
intlist = [0,1,2,3,4,5,6,7,8,9]
heterogeniouslist = [1,"two", True]
listoflist = [intlist, heterogeniouslist]
print(listoflist)
print(f"Last element of list {intlist[-1]} ")
print(f"First element of list {intlist[0]} ")
print(f"First 3 elements of list {intlist[:3]} ") #read as range from 0 to 3, strictly less than upper bound
print(f"Last 3 elements of list {intlist[-3:]} ") #read as -3 (third last) to end
print(f"Sub list {intlist[2:6]} ") #read as  
print(f"Sub list {intlist[::4]} ") #read as - take every 4th element from the entire list 
print(f"Sub list {intlist[:6:2]} ") #read as  - take every 2nd element from the list [0,6)
#%%
x,y = [10,20]
print(f"x={x}, y={y}") #unpack list to finite elements
#%%
#Tuple - immutable list
tup= (2,"pop")
tup1 = 3,4
print(tup, tup1, tup1[-1])
x, y = 1 , 2
print(f"x={x}, y={y}")
x,y=y,x 
print(f"x={x}, y={y}")
a=10
b=20
a,b = b,a
print(f"a-{a} , b-{b}")
#%%
q=10
w=20
q=q+w
w=q-w
q=q-w
print(f"q-{q} , w-{w}")
#%%
#Dictionary
empty_dict = {}
empty_dict1 = dict()
grades = {"Joel": 80, "Tim": 95}

for gr in grades:
    print(f"key - {gr} , value - {grades[gr]}")
print(f"Joel's grade {grades.get('Joel')}")
#%%
#Default dict. - When you want to initialize mising key with some default value. Perfect for word count program
import collections as col 
defdict = col.defaultdict(int) # read as dictionary with value as integer
print(defdict["hi"]) 

dd_list = col.defaultdict(list)             # list() produces an empty list with value as list can key can be anything
dd_list[2].append(1)                    # now dd_list contains {2: [1]}
print(dd_list[2])
dd_dict = col.defaultdict(dict)             # dict() produces an empty dict
dd_dict["Joel"]["City"] = "Seattle"     # {"Joel" : {"City": Seattle"}}
dd_pair = col.defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1                       # now dd_pair contains {2: [0, 1]}

#Counter
ctr = col.Counter([1,2,3,4,3,2,2,3,3,4,4,5,3,3])
print(ctr)
print(ctr.most_common(3))

#%%
#Sets
primes_below_10 = {2, 3, 5, 7} # for empty set use set() since {} means dictionary. Used for 2 main reasons 
                               # 1) in operation is fast
                               # 2) Find distinct items in lit 
s= {1,2,3,1,3,4}
print( type(s))
print( type([2,3,4,5]))
print(list(s))
#%%
#Truthiness
print(any([])) # False since no true element in list
print(all([])) # True since no false element in list 
#%%
#List Comprehensions
evenNumbers = [x for x in range(6) if x%2==0]
print(f"Even number - {evenNumbers} ")
square = [x*x for x in [4,5,6]]
print(f"square number - {square} ")
allZeros = [0 for _ in evenNumbers]
print(f"All Zeros - {allZeros} ")
pairs = [(x,y) 
        for x in range(10) if x%2==0 #get even numbers 
        for y in [2,4]
        ]
print (f"Pairs {pairs}")
#%%
#class
class CountingClicker:
    def __init__(self, count=1):
        self.Count = count
    def Click(self,numOfTimes):
        self.Count += numOfTimes
    def ShowClicks(self):
        print(self.Count)
cl = CountingClicker(4)
cl.ShowClicks()
cl.Click(10)
cl.ShowClicks()

#%%
#Iterables and Generators
names = ["Alice", "Bob", "Charlie", "Debbie"]
i=0
#non pythonic way
for name in names:
    print(f"i - {i}, name - {name}")
    i+=1
# pythonic way
for index, name in enumerate(names):
    print(f"index - {index}, name - {name}")
#%%
#zip and Argument Unpacking
list1 = ['a', 'b', 'c']
list2 = [1, 2,3,4]
pairs = zip(list1, list2)
print(pairs)
print([pair for pair in zip(list1, list2)] ) #zip
l1 , l2 = zip(*pairs) #unzip
print(l1)
print(l2)
def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)
magic(1, 2, key="word", key2="word2")


# %%
#Understand map function
def cube(num):
    """
    It returns cube of number
    """
    return num**3

lst =list(range(0,10))
print(len(lst))
#try to get cube of every number in list using cube function
result = [cube(num) for num in lst]
print(result)
#use map
mapresult = list(map(cube,lst))
print(mapresult)
#Use lambda expression with map
lambdaresult = list(map(lambda num:num**3, lst))
print(lambdaresult)
#filter 
filtered = list(filter(lambda num:num%2==0, lst))
print(filtered)
# %%
