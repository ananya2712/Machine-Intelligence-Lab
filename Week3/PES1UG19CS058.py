'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float

def get_entropy_of_dataset(df):
    # TODO
    entropy=0
    if df.empty:
        return entropy
    last_column = df.iloc[: , -1]
    #return one coloumn with unique elements and one with their counts
    ele,counts = np.unique(last_column,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(ele))])
    return entropy

def get_entropy_of_attribute(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    """entropy=0
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy"""
    


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    # TODO
    #obtain unique values in attribute
    """ele,counts = np.unique(df[attribute],return_counts = True)
    
    avg_info = np.sum([(counts[i]/np.sum(counts))*get_entropy_of_attribute(df.where(df[attribute]==ele[i])) for i in range(len(ele))])
    return avg_info"""

    entropy_of_attribute = 0
    value, counts = np.unique(df[attribute], return_counts=True)
    norm_counts = counts / counts.sum()
    base = 2
    entropy = np.array([])
    for i in range(0,len(value)):
        split = df.loc[df[attribute] == value[i]]
        val, cnts = np.unique(split.iloc[:,-1], return_counts=True)
        norm_cnts = cnts / cnts.sum()
        entropy = np.append(entropy,-(norm_cnts * np.log(norm_cnts) / np.log(base)).sum())
    entropy_of_attribute = (norm_counts * entropy).sum()
    return abs(entropy_of_attribute)


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    # TODO
    info_gain = 0
    entropy = get_entropy_of_dataset(df)

    info_gain = entropy - get_avg_info_of_attribute(df,attribute)
    return info_gain



#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    # TODO
    information_gains=dict()
    selected_column=selected_column_max_value=None

    for column_name in list(df.columns)[: -1]:
        information_gains[column_name] = get_information_gain(df, column_name)
        if selected_column_max_value == None or selected_column_max_value < information_gains[column_name]:
            selected_column_max_value = information_gains[column_name]
            selected_column = column_name

    return (information_gains,selected_column)

