#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt

from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval

pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class FeatureTransform():
    def __init__(self, df):
        self.df = df
        self.feature_funcs = {'conti':self.transform_conti_data, 'count':self.transform_count_data,
                     'scale':self.transform_scale_data, 'bin':self.transform_bin_data,
                     'log':self.transform_log_data, 'nominal':self.transform_nominal_data,
                     'ordinal':self.transform_ordinal_data, 'hash': self.transform_hash_data,
                     'ordinaltolabel':self.transform_ordinal_data_label, 'multiLabel':self.transform_multilabel_data,
                     'oneHot':self.transform_onehot_data, 'transtolist':self.transform_to_listdata, 'default':self.show_dataframe}
    
    def wrap(pre, post):
        """ Wrapper """
        def decorate(func):
            """ Decorator """
            def call(*args, **kwargs):
                """ Actual wrapping """
                pre(func)
                result = func(*args, **kwargs)
                post(func)
                return result
            return call
        return decorate
    
    def entering(func):
        """ Pre function logging """
        # print("Entered: ", func.__name__)
        pass

    
    def exiting(func):
        """ Post function logging """
        # print("Exited: ", func.__name__)
        pass

    @wrap(entering, exiting)    
    def transform_conti_data(self, **kwargs):
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr
    
    @wrap(entering, exiting)
    def transform_count_data(self, **kwargs):
        feature_arr = self.df[kwargs['column_name']] > kwargs['threshold'] 
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr
    
    @wrap(entering, exiting)
    def transform_scale_data(self, **kwargs):
        feature_arr = self.df[kwargs['column_name']]/kwargs['factor']
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr
    
    @wrap(entering, exiting)
    def transform_bin_data(self, **kwargs):
        
        enc = KBinsDiscretizer(kwargs['n_bins'], kwargs['encode'], kwargs['strategy'])
        feature_arr_enc = enc.fit_transform(self.df[[kwargs['column_name']]].dropna()).flatten()
        #print (type(feature_arr_enc))
        feature_arr = pd.cut(self.df[kwargs['column_name']], np.concatenate(enc.bin_edges_, axis=0), right=False)
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr

    @wrap(entering, exiting)    
    def transform_log_data(self, **kwargs):
        feature_arr = np.log(self.df[kwargs['column_name']])
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr
    
    
    @wrap(entering, exiting)
    def transform_nominal_data(self, **kwargs):
        enc = OneHotEncoder(categories='auto', handle_unknown='ignore').fit(self.df[[kwargs['column_name']]].dropna())
        print (enc.categories_)
        feature_arr = enc.transform(self.df[[kwargs['column_name']]].replace([np.nan], [-1])).toarray()
        # print (feature_arr)
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr
        
    @wrap(entering, exiting)
    def transform_ordinal_data_label(self, **kwargs):
        feature_arr = self.df[kwargs['column_name']].map(kwargs['order'])
        self.add_column_df(kwargs['column_name'], feature_arr, '-Label')
        return feature_arr
        
    @wrap(entering, exiting)
    def transform_ordinal_data(self, **kwargs):
        enc = OneHotEncoder(categories='auto', handle_unknown='ignore').fit(self.df[[kwargs['column_name']]].dropna())
        feature_arr = enc.fit_transform(self.df[[kwargs['column_name']]].replace([np.nan], [[]])).toarray()
        
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr
    
    @wrap(entering, exiting)
    def transform_hash_data(self, **kwargs):
        enc = FeatureHasher(n_features=kwargs['n_features'], input_type='string')
        feature_arr = enc.fit_transform(
                              self.df[kwargs['column_name']]).toarray()
        #print (feature_arr)
        self.add_column_df(kwargs['column_name'], feature_arr)
        return feature_arr
    
    @wrap(entering, exiting)
    def transform_to_listdata(self, **kwargs):
        return feature_arr
    
    @wrap(entering, exiting)
    def transform_multilabel_data(self, **kwargs):
        #feature_arr = self.feature_funcs[kwargs['data_type_func']](**kwargs)
        #feature_arr = []
        mlb = MultiLabelBinarizer(classes=kwargs['classes'])
        if kwargs['literal'] == True:
            mlb.fit(self.df[kwargs['column_name']].dropna().apply(literal_eval))
            feature_arr = mlb.transform(self.df[kwargs['column_name']].replace([np.nan], ['[]']).apply(literal_eval))
        else:
            mlb.fit(self.df[kwargs['column_name']].dropna())
            feature_arr = mlb.transform(self.df[kwargs['column_name']].replace([np.nan], ['[]']))
        self.add_column_df(kwargs['column_name'], feature_arr)
        tf_df = pd.DataFrame(feature_arr, columns=[kwargs['column_name']+'_TFV_'+cls for cls in mlb.classes_],
                             index=self.df.index)
        
        ## Todo change columns name
        #tf_df = tf_df[[kwargs['column_name']+'_'+col+'_TRV_' for col in tf_df.columns.tolist()]]
        self.add_column_after(kwargs['column_name']+'_TF_', tf_df)
        
        return tf_df
    
    @wrap(entering, exiting)
    def transform_onehot_data(self, **kwargs):
        feature_arr = self.feature_funcs[kwargs['data_type_func']](**kwargs)
        
        if kwargs['data_type_func'] == 'bin':
            print ('bin data')
            tf_df = pd.get_dummies(data=pd.DataFrame(data=feature_arr, columns=[kwargs['column_name']]), prefix_sep='_TFV_')
        elif(self.df[[kwargs['column_name']]].dtypes.iloc[0]==np.bool):
            print('boolean data')
            tf_df = pd.get_dummies(data=self.df[[kwargs['column_name']]].replace([False, True], ['False', 'True']), prefix_sep='_TFV_')
        else:
            tf_df = pd.get_dummies(data=self.df[[kwargs['column_name']]], prefix_sep='_TFV_')
        
        
        self.add_column_after(kwargs['column_name']+'_TF_', tf_df)
        
        #return feature_arr
    
    @wrap(entering, exiting)
    def getFile(self, **kwargs):
        return self.file
    
    @wrap(entering, exiting)
    def plotData(self,plottype, **kwargs):
        pass
    
    
    @wrap(entering, exiting)
    def get_data_frame(self):
        return self.df
    
    @wrap(entering, exiting)
    def set_data_frame(self, df):
        self.df = df
    
    
    @wrap(entering, exiting)
    def add_column_df(self, column_name, feature_arr, suffix='_TF_'):
        #feature_arr_T = feature_arr.transpose()
        #[self.df.insert(loc=df.columns.get_loc(column_name)+1+i, column=column_name[0:2]+str(i), value=new_col) 
                                                #for i, new_col in zip(range(len(feature_arr_T)), feature_arr_T)]
        self.df.insert(loc=self.df.columns.get_loc(column_name)+1, column=column_name+suffix, value=feature_arr.tolist())
        #self.df = self.df.assign(e=feature_arr)
    
    @wrap(entering, exiting)
    def add_column_after(self, column_name, tf_df):
        pre_df = self.df.iloc[:,0:self.df.columns.get_loc(column_name)+1]
        post_df =self.df.iloc[:,self.df.columns.get_loc(column_name)+1:]
        tf_df[pd.isna(pre_df.iloc[:,-2])]=np.nan
        self.df = pd.concat([pre_df, tf_df, post_df], axis=1)

    
    @wrap(entering, exiting)
    def show_dataframe(self, n=5):
        #print(self.df.head(n))
        display(self.df.head(n))
        
    
    @wrap(entering, exiting)    
    def extract_value_frm_list(self, x):
        clss = set()
        for l in list(x):
            try:
                for ll in list(l):
                    clss.add(str(ll))
            except Exception as e:
                print ('', end='')
        #print (clss)        
        return list(clss)
        
        
    @wrap(entering, exiting)
    def apply_feature_transform(self, **kwargs):
        '''try:
            print('Transformation Column Name: {}, Function Type: {}'.format(kwargs['column_name'],kwargs['func_type']))
            feature_funcs[kwargs['func_type']](**kwargs)
        except:
            print('Error: No Feature Transformation')
            feature_funcs['default']'''
        self.feature_funcs[kwargs['func_type']](**kwargs)
        return self.get_data_frame()


# In[ ]:




