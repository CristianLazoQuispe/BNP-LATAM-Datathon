import pandas as pd
import numpy as np
from IPython.display import display, HTML


def my_df_describe(df,name = 'dataset',show = True):
    print(20*'*'+name+20*'*')
    objects = []
    numerics = []
    for c in df:
        if (df[c].dtype == object):
            objects.append(c)
        else:
            numerics.append(c)
    
    df = df.replace(to_replace=['',' ','None','NaN'], value=np.nan)
    numeric_desc, categorical_desc = None,None

    if len(numerics)>0:
        numeric_desc = df[numerics].describe().transpose()
        counting = df.nunique().transpose()
        numeric_desc['unique'] = counting[numerics]
        numeric_desc['nulls'] = df[numerics].isna().sum().values
        numeric_desc['nulls_perc'] = df[numerics].isna().sum().values/df.shape[0]
        #print(df[numerics].isna().sum())
        #print(numeric_desc.shape)

    if len(objects)>0:
        categorical_desc = df[objects].describe().transpose()
        #print(df[objects].isna().sum())
        #print(categorical_desc.shape)
        categorical_desc['nulls'] = df[objects].isna().sum().values
        categorical_desc['nulls_perc'] = df[objects].isna().sum().values/df.shape[0]
    
    print('shape ',df.shape)
    if show:
        if len(numerics)>0:
            print(10*'*'+'numerics'+10*'*')
            display(numeric_desc)
        if len(objects)>0:
            print(10*'*'+'categorical'+10*'*')
            display(categorical_desc)

    return numeric_desc, categorical_desc

def clean_null(df_train,df_test,limit = 0):
    print('Train init shape ',df_train.shape)
    print('Test  init shape ',df_test.shape)
    df_train = df_train.replace(to_replace=['None','NaN'], value=np.nan)
    
    '''
    columns_ok = df_train.notna().any(axis=0)
    print('columns_ok ',columns_ok.sum())
    print('columns_no_ok ',len(df_train.columns)-columns_ok.sum())
    for name in df_train.columns:
        if not columns_ok[name]:
            print('has 0 elements -> deleting ..',name)
    df_train = df_train.loc[:,columns_ok]
    df_test = df_test.loc[:,columns_ok]
    '''
    columms_percent_nulls = df_train.isnull().mean()
    columns_ok = columms_percent_nulls < limit
    for name in df_train.columns:
        if not columns_ok[name]:
            print('has ',100*columms_percent_nulls[name] ,"%"+" null elements -> deleting ..",name)

    df_train = df_train.loc[:,columns_ok]
    df_test = df_test.loc[:,columns_ok]

    print('Train final shape',df_train.shape)
    print('Test  final shape',df_test.shape)
    
    return df_train,df_test

