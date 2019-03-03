import numpy as np
import pandas as pd

def my_gini(panda, col):
    grouping = panda.groupby(col)[col]
    prob_array = grouping.count()/panda[col].count()
    p = 0
    names = []
    for name,_ in grouping:
        names.append(name)
    for i in range(len(prob_array)):
        p+=np.square(prob_array[names[i]])
    return 1 - p

def weighted_gini(parent, child, col):
    weight = child[col].count()/parent[col].count()
    return weight*my_gini(child, col)

def my_entropy(panda, col):
    p = 0
    names = []
    grouping = panda.groupby(col)[col]
    pb = grouping.count()/panda[col].count()
    for name, _ in grouping:
        names.append(name)
    for i in range(len(names)):
        p+= pb[names[i]]*np.log2(pb[names[i]])
    return -p

def information_gain(panda, right, left, col):
    child_ent = (right[col].count()/panda[col].count())*my_entropy(right, col) + (left[col].count()/panda[col].count())*my_entropy(left, col)
    return my_entropy(panda, col) - child_ent

def my_misclass(panda, col):
    p = 0
    names = []
    grouping = panda.groupby(col)[col]
    prob_array = grouping.count()/panda[col].count()
    return 1 - max(prob_array)

def weighted_misclass(parent, child, col):
    weight = child[col].count()/parent[col].count()
    return weight*my_misclass(child, col)
