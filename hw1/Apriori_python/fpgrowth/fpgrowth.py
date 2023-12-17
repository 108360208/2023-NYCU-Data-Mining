# mlxtend Machine Learning Library Extensions
# Author: Steve Harenberg <harenbergsd@gmail.com>
#
# License: BSD 3 clause

import itertools
import math
from optparse import OptionParser
import pandas as pd
import time
import os
from transactionencoder import TransactionEncoder
import fpcommon as fpc


def fpgrowth(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0):
    """Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. Also supports
      DataFrames with sparse data; for more info, please
      see https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html#sparse-data-structures.

      Please note that the old pandas SparseDataFrame format
      is no longer supported in mlxtend >= 0.17.2.

      The allowed values are either 0/1 or True/False.
      For example,

    ```
           Apple  Bananas   Beer  Chicken   Milk   Rice
        0   True    False   True     True  False   True
        1   True    False   True    False  False   True
        2   True    False   True    False  False  False
        3   True     True  False    False  False  False
        4  False    False   True     True   True   True
        5  False    False   True    False   True   True
        6  False    False   True    False   True  False
        7   True     True  False    False  False  False
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minimum support of the itemsets returned.
      The support is computed as the fraction
      transactions_where_item(s)_occur / total_transactions.

    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths are evaluated.

    verbose : int (default: 0)
      Shows the stages of conditional tree generation.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    ----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/

    """
    fpc.valid_input_check(df)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )

    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, _ = fpc.setup_fptree(df, min_support)
    minsup = math.ceil(min_support * len(df.index))  # min support as count
    generator = fpg_step(tree, minsup, colname_map, max_len, verbose)

    return fpc.generate_itemsets(generator, len(df.index), colname_map)


def fpg_step(tree, minsup, colnames, max_len, verbose):
    """
    Performs a recursive step of the fpgrowth algorithm.

    Parameters
    ----------
    tree : FPTree
    minsup : int

    Yields
    ------
    lists of strings
        Set of items that has occurred in minsup itemsets.
    """
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    if verbose:
        tree.print_status(count, colnames)

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(cond_tree, minsup, colnames, max_len, verbose):
                yield sup, iset
def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    result = []
    with open(fname, "r") as file_iter: 
        for line in file_iter:
            line = line.strip(' ').rstrip("\n")  # Remove trailing comma        
            record = (line.split(" ")[3:])
            result.append(record)
        return result

def custom_format(row):
    support = row['support'] * 100
    itemset = "{" + ",".join(map(str, row['itemsets'])) + "}"
    return f"{support:.1f}\t{itemset}\n"

def writeTask1File(output_file, dataset_file, min_support, items):
    if not os.path.exists(output_file):
    # 如果資料夾不存在，則使用os.makedirs創建它
        os.makedirs(output_file)
        print(f"資料夾 '{output_file}' 已創建")
    else:
        print(f"資料夾 '{output_file}' 已存在")

    input_string = dataset_file
    data_index = input_string.find(".data")
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True) 
    if data_index != -1:
    # 如果找到了 ".data"，提取它前面的英文字母
        dataset = input_string[data_index - 1]
    else:
        print("没有找到 '.data'")

    print("\033[101;97m" +"Start write Task1 result1 file!!!" + "\033[0m")
    with open(os.path.join(output_file, f"step3_task1_dataset({dataset})_{min_support}_result1.txt"), "w") as file:
      for i in items['formatted']:
          file.write(i)
    print("\033[47;30m"+"Write file end!!!"+ "\033[0m")







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
    dataset = dataFromFile(options.input)
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
  
    
    print("\033[101;97m" +"Start Mining!!!" + "\033[0m")
    start = time.time()
    result = fpgrowth(df, options.minS, use_colnames=True)
    end = time.time()
    print("\033[47;30m"+"End Mining!!!"+ "\033[0m")
    print(f"Count the computation time for task1 is {end - start}s") 
    result['formatted'] = result.apply(custom_format, axis=1)
    result = result.sort_values(by='support', ascending=False)
    writeTask1File(options.outputFilePath, options.input, options.minS, result)