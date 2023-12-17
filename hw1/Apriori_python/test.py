import numpy as np
import csv
from datetime import datetime
from math import ceil
from bitarray import bitarray
from optparse import OptionParser
import time
import os

class BMCTreeNode:
    def __init__(self, item, count, bitmap_code):
        self.item = item
        self.count = count
        self.bitmap_code = bitmap_code
        self.children = dict()

    def get_child_registering_item(self, item):
        return self.children.get(item)

    def add_child(self, child):
        self.children[child.item] = child

    def __repr__(self):
        return f'{self.item}:{self.count}->{self.bitmap_code}'


class FrequentItemsetTreeNode:
    def __init__(self):
        self.item = None
        self.count = 0
        self.children = []
        self.NegNodeSet = []

    def __repr__(self):
        return f'{self.item}'

def clean_BMC_tree(root):
    for item, child in root.children.items():
        clean_BMC_tree(child)
    del root.item
    del root.children

class NegFIN:
    def __init__(self, dataset_file, min_support, output_file, delimiter=' '):
        self.dataset_file = dataset_file
        self.min_support = min_support
        self.min_count = None
        self.output_file = output_file
        self.delimiter = delimiter
        self.num_of_transactions = None
        self.F1 = None
        self.item_to_NodeSet = None
        self.writer = None
        self.num_of_frequent_itemsets = 0
        self.execution_time = None
        self.file_buffer = []

    def __find_F1(self):
        item_name_to_count = {}
        with open(self.dataset_file) as file_input:
            reader = csv.reader(file_input, delimiter=self.delimiter)
            self.num_of_transactions = 0
            for transaction in reader:
                self.num_of_transactions += 1
                for item_name in transaction[3:]:
                    item_count = item_name_to_count.setdefault(item_name, 0)
                    item_name_to_count[item_name] = item_count + 1
        item_name_to_count.pop('', None)
        self.min_count = ceil(self.num_of_transactions * self.min_support)
        self.F1 = [{'name': item_name, 'count': item_count} for (item_name, item_count) in item_name_to_count.items() if
                   self.min_count <= item_count]
        self.F1.sort(key=lambda item: item['count'])

        self.F1 = np.array([(item['name'], item['count']) for item in self.F1], dtype=[('name', 'U20'), ('count', int)])


    def __generate_NodeSets_of_1_itemsets(self):
        item_name_to_item_index = {item['name']: item_index for (item_index, item) in enumerate(self.F1)}
        self.item_to_NodeSet = {item_index: [] for item_index in item_name_to_item_index.values()}
        bmc_tree_root = BMCTreeNode(item=None, count=None, bitmap_code=bitarray([False] * len(self.F1)))

        with open(self.dataset_file) as fInput:
            reader = csv.reader(fInput, delimiter=self.delimiter)
            for transaction in reader:
                transaction = [item_name_to_item_index[item_name] for item_name in transaction[3:] if
                               item_name in item_name_to_item_index]
                transaction.sort(reverse=True)
                cur_root = bmc_tree_root
                for item in transaction:
                    N = cur_root.get_child_registering_item(item)
                    if N is None:
                        bitmap_code = cur_root.bitmap_code.copy()
                        bitmap_code[item] = True
                        N = BMCTreeNode(item=item, count=0, bitmap_code=bitmap_code)
                        cur_root.add_child(N)
                        self.item_to_NodeSet[item].append(N)

                    N.count += 1
                    cur_root = N

        clean_BMC_tree(bmc_tree_root)

    def __create_root_of_frequent_itemset_tree(self):
        root = FrequentItemsetTreeNode()

        for item in range(len(self.F1)):
            child = FrequentItemsetTreeNode()
            child.item = item
            child.count = self.F1[item]['count']
            child.NegNodeSet = self.item_to_NodeSet[item]
            root.children.append(child)
        return root

    def __write_itemsets_to_file(self, N, itemset_buffer, N_itemset_length, FIS_parent_buffer, FIS_parent_length):
        file_buffer = []
        self.num_of_frequent_itemsets += 1
        itemset_string = [self.F1[itemset_buffer[i]]['name'] for i in range(N_itemset_length)]

        sorted_items = sorted(self.file_buffer, key=lambda x: x[1], reverse=True)

        if FIS_parent_length > 0:
            max = 1 << FIS_parent_length
            for i in range(1, max):
                itemset_string = [self.F1[itemset_buffer[i]]['name'] for i in range(N_itemset_length)]
                subsetString = [self.F1[FIS_parent_buffer[j]]['name'] for j in range(FIS_parent_length) if (i & (1 << j)) > 0]
                itemset_string.extend(subsetString)
                self.file_buffer.extend([(tuple(itemset_string), N.count)])
                self.num_of_frequent_itemsets += 1

    def __construct_frequent_itemset_tree(self, N, itemset_buffer, N_itemset_length, N_right_siblings, FIS_parent_buffer,
                                          FIS_parent_length):
        for sibling in N_right_siblings:
            child = FrequentItemsetTreeNode()
            sum_of_NegNodeSets_counts = 0
            if N_itemset_length == 1:
                for ni in N.NegNodeSet:
                    if not ni.bitmap_code[sibling.item]:
                        child.NegNodeSet.append(ni)
                        sum_of_NegNodeSets_counts += ni.count
            else:
                for nj in sibling.NegNodeSet:
                    if nj.bitmap_code[N.item]:
                        child.NegNodeSet.append(nj)
                        sum_of_NegNodeSets_counts += nj.count
            child.count = N.count - sum_of_NegNodeSets_counts
            if self.min_count <= child.count:
                if N.count == child.count:
                    FIS_parent_buffer[FIS_parent_length] = sibling.item
                    FIS_parent_length += 1
                else:
                    child.item = sibling.item
                    N.children.append(child)

        self.__write_itemsets_to_file(N, itemset_buffer, N_itemset_length, FIS_parent_buffer, FIS_parent_length)

        number_of_childeren = len(N.children)
        for childIndex in range(number_of_childeren):
            child = N.children[0]
            itemset_buffer[N_itemset_length] = child.item
            del N.children[0]
            self.__construct_frequent_itemset_tree(child, itemset_buffer, N_itemset_length + 1, N.children,
                                                   FIS_parent_buffer, FIS_parent_length)

    def runAlgorithm(self):
        start_time = time.time()
        self.__find_F1()
        self.__generate_NodeSets_of_1_itemsets()
        root = self.__create_root_of_frequent_itemset_tree()
        itemset_buffer = [None] * len(self.F1)
        itemset_length = 0
        FIS_parent_buffer = [None] * len(self.F1)
        FIS_parent_length = 0
        num_of_children = len(root.children)

        for childIndex in range(num_of_children):
            child = root.children[0]
            itemset_buffer[itemset_length] = child.item
            del root.children[0]
            self.__construct_frequent_itemset_tree(child, itemset_buffer, itemset_length + 1, root.children,
                                                   FIS_parent_buffer, FIS_parent_length)

        end_time = time.time()
        time_diff = (end_time - start_time)
        self.execution_time = time_diff
        sorted_items = sorted(self.file_buffer, key=lambda x: x[1], reverse=True)
        self.writeTask1File(sorted_items)

    def printStats(self):
        print('=' * 5 + 'negFIN - STATS' + '=' * 5)
        print(f' Minsup = {self.min_support}\n Number of transactions: {self.num_of_transactions}')
        print(f' Number of frequent itemsets: {self.num_of_frequent_itemsets}')
        print(f' Total time : {self.execution_time}s')
        print('=' * 14)

    def writeTask1File(self, items):
        if not os.path.exists(self.output_file):
            os.makedirs(self.output_file)
            print(f"Folder '{self.output_file}' has been created")
        else:
            print(f"Folder '{self.output_file}' already exists")

        input_string = self.dataset_file
        data_index = input_string.find(".data")
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        if data_index != -1:
            dataset = input_string[data_index - 1]
        else:
            print("'.data' not found")

        print("\033[101;97m" + "Start writing Task1 result1 file!!!" + "\033[0m")
        with open(os.path.join(self.output_file, f"step3_task1_dataset({dataset})_{self.min_support}_result1.txt"), "w") as file:
            for i in sorted_items:
                item, support = i
                itemset = "{" + ",".join(item) + "}"
                line = f"{support / self.num_of_transactions * 100:.1f}\t{itemset}\n"
                file.write(line)
        print("\033[47;30m" + "Writing file end!!!" + "\033[0m")

if __name__ == '__main__':
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

    (options, args) = optparser.parse_args()

    delimiter1 = ' '
    algorithm = NegFIN(options.input, options.minS, options.outputFilePath, delimiter1)
    print("\033[101;97m" + "Start Mining!!!" + "\033[0m")
    algorithm.runAlgorithm()
    print("\033[47;30m" + "End Mining!!!" + "\033[0m")
    algorithm.printStats()
