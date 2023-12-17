"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python apriori.py -f DATASET.csv -s minSupport

    $python apriori.py -f DATASET.csv -s 0.15
"""

import sys
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import os
def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

# 得到itemset的組合
def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
 
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets

    return itemSet, transactionList


def runApriori(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    start_time = time.time()
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    resultFileTwoList = []
    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    oneCSet= returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    resultFileTwoList.append(f"{1}\t{len(itemSet)}\t{len(oneCSet)}\n")
    currentLSet = oneCSet

    k = 2
    while currentLSet != set([]):    
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet= returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        
        resultFileTwoList.append(f"{k}\t{len(currentLSet)}\t{len(currentCSet)}\n")
        currentLSet = currentCSet
        k = k + 1
    print(len(largeSet))
    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)
    # print(freqSet)
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])
  
    resultFileTwoList.insert(0, f"{len(toRetItems)}\n")

    task1_end_time = time.time()

    closedItemset = []
    for key, value in largeSet.items():
        if(key != len(largeSet)):
            for sub_item in value:
                Closed = True
                for super_item in largeSet[key + 1]:
                    if(set(sub_item).issubset(set(super_item)) and getSupport(sub_item) == getSupport(super_item)):
                        Closed = False
                        break
                if (Closed):
                    closedItemset.append((tuple(sub_item), getSupport(sub_item)))
        else:
            for super_item in largeSet[key]:
                closedItemset.append((tuple(super_item), getSupport(super_item)))
              
    task2_end_time = time.time()
    task1Time = task1_end_time - start_time   
    task2Time = task2_end_time - start_time      
    ratio = (task2Time / task1Time) * 100
    print(f"Count the computation time for task1 is {task1Time}s") 
    print(f"Count the computation time for task2 is {task2Time}s") 
    print(f"The ratio of computation time compared to that of Task 1 is {ratio}%")   

    return toRetItems, resultFileTwoList, closedItemset

from collections import defaultdict

# def runApriori_closed(data_iter, minSupport):
#     """
#     run the apriori algorithm to find Frequent Closed Itemsets.
#     data_iter is a record iterator.
#     Return a list of frequent closed itemsets in the form (items, support).
#     """
#     itemSet, transactionList = getItemSetTransactionList(data_iter)

#     freqSet = defaultdict(int)
#     closedSet = defaultdict(int)
    
#     oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

#     currentLSet = oneCSet
#     k = 2
    
#     while currentLSet != set([]):
#         closedSet[k - 1] = currentLSet
#         currentLSet = joinSet(currentLSet, k)
#         currentCSet = returnItemsWithMinSupport(currentLSet, transactionList, minSupport, freqSet)
#         currentLSet = currentCSet
#         k = k + 1

#     def isClosed(itemset, closedSet):
#         """
#         Checks if the itemset is a frequent closed itemset by comparing it with other frequent itemsets.
#         """
#         for key, value in closedSet.items():
#             for i in value:
#                 if set(itemset) != set(i) and set(itemset).issubset(set(i)) and freqSet[itemset] == freqSet[i]:
#                     return False
#         return True
#     def getSupport(item):
#         """local function which Returns the support of an item"""
#         return float(freqSet[item]) / len(transactionList)
#     toRetItems = []
#     for key, value in closedSet.items():
#         for item in value:
#             if isClosed(item, closedSet):
#                 toRetItems.extend([(tuple(item), getSupport(item))])
                

#     return toRetItems



        
def printResults(items):
    """prints the generated itemsets sorted by support """
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))


def to_str_results(items):
    """prints the generated itemsets sorted by support"""
    i = []
    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)
    return i

def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter: 
        for line in file_iter:
            line = line.strip(' ').rstrip("\n")  # Remove trailing comma        
            record = frozenset(line.split(" ")[3:])
            yield record
    

def writeTask1File(options,items, resultFileTwoList):
    if not os.path.exists(options.outputFilePath):
    # 如果資料夾不存在，則使用os.makedirs創建它
        os.makedirs(options.outputFilePath)
        print(f"資料夾 '{options.outputFilePath}' 已創建")
    else:
        print(f"資料夾 '{options.outputFilePath}' 已存在")

    input_string = options.input
    data_index = input_string.find(".data")
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True) 
    if data_index != -1:
    # 如果找到了 ".data"，提取它前面的英文字母
        dataset = input_string[data_index - 1]
    else:
        print("没有找到 '.data'")

    print("\033[101;97m" +"Start write Task1 result1 file!!!" + "\033[0m")
    with open(os.path.join(options.outputFilePath, f"{options.step}_task1_dataset({dataset})_{options.minS}_result1.txt"), "w") as file:
        for i in sorted_items:
            item, support = i          
            itemset = "{" + ",".join(item) + "}"
            line = f"{support*100:.1f}\t{itemset}\n"
            file.write(line)
    print("\033[47;30m"+"Write file end!!!"+ "\033[0m")


    print("\033[101;97m" +"Start write Task1 result2 file!!!" + "\033[0m")
    with open(os.path.join(options.outputFilePath, f"{options.step}_task1_dataset({dataset})_{options.minS}_result2.txt"), "w") as file:
        for i in resultFileTwoList:
            file.write(i)
    print("\033[47;30m"+"Write file end!!!"+ "\033[0m")


def writeTask2File(options, closedItemsetList):
    if not os.path.exists(options.outputFilePath):
    # 如果資料夾不存在，則使用os.makedirs創建它
        os.makedirs(options.outputFilePath)
        print(f"資料夾 '{options.outputFilePath}' 已創建")
    else:
        print(f"資料夾 '{options.outputFilePath}' 已存在")
    sorted_items = sorted(closedItemsetList, key=lambda x: x[1], reverse=True) 
    input_string = options.input
    data_index = input_string.find(".data")
    if data_index != -1:
    # 如果找到了 ".data"，提取它前面的英文字母
        dataset = input_string[data_index - 1]
    else:
        print("没有找到 '.data'")

    print("\033[101;97m" +"Start write Task2 result1 file!!!" + "\033[0m")
    with open(os.path.join(options.outputFilePath, f"step2_task2_dataset({dataset})_{options.minS}_result1.txt"), "w") as file:
            file.write(f"{len(sorted_items)}\n")
            for i in sorted_items:
                itemSet, support = i
                itemset = "{" + ",".join(itemSet) + "}"
                line = f"{support*100:.1f}\t{itemset}\n"
                file.write(line)
    print("\033[47;30m"+"Write file end!!!"+ "\033[0m")

if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="filename containing csv", default='formatted_data.csv'
    )
    optparser.add_option(
        "-o", "--outputFilePath", dest="outputFilePath", help="the root of result file", default=''
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.003,
        type="float",
    )
    optparser.add_option(
        "-t",
        "--step",
        dest="step",
        help="step 2 or step 3",
        default="step2",
        type="string",
    )
    
    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")
   
    minSupport = options.minS
   
    print("\033[101;97m" +"Start Mining!!!" + "\033[0m")
    
    items, resultFileTwoList, closedItemsetList = runApriori(inFile, minSupport)

    print("\033[47;30m"+"End Mining!!!"+ "\033[0m")
    writeTask1File(options,items, resultFileTwoList)
    writeTask2File(options, closedItemsetList)

    # closedItemsetList = runApriori_closed(inFile, minSupport)
    # writeTask2File(options, closedItemsetList)
