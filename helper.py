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
