#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '../preprocessing/')
import numpy as np
import pickle
import scipy.stats as spstats
import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas_profiling
from sklearn.utils import Bunch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, make_scorer
import re

import pandas as pd
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

from ordered_set import OrderedSet

from func_def import *

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# data from variable guide
data_variable_cat = {}
with open("data_variable_cat.pkl", "rb") as f:
    data_variable_cat = pickle.load(f)

len(data_variable_cat)


# In[3]:


df_data_comp = pd.read_pickle(orginal_data_path+'Optima_Data_Report_Cases_9584_filled_pickle')
df_data_comp.sort_values(by=['GLOBAL_PATIENT_DB_ID', 'EPISODE_DATE'], inplace=True)
df_data_comp.head(1)


# In[4]:


# Analysis Recall Objects
# Rename these columns
df_data_comp.rename(columns={'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS':'COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS_3', 
                             'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS':'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS_6'}, inplace=True)

df_data_comp[['COGNITIVE EXAM 120-161: (161) RECALLS OBJECTS_3', 'COGNITIVE EXAM 120-161: (146) RECALLS OBJECTS_6']].hist()


# In[5]:


df_data_comp['durations(years)'] = df_data_comp.groupby(by='GLOBAL_PATIENT_DB_ID')['EPISODE_DATE'].transform(lambda x: (x - x.iloc[0])/(np.timedelta64(1, 'D')*365.25))
df_data_comp['MINI MENTAL SCORE PRE'] = df_data_comp.groupby(by='GLOBAL_PATIENT_DB_ID')['CAMDEX SCORES: MINI MENTAL SCORE'].transform(lambda x: x.shift(+1))


# In[6]:


misdiagnosed_df = pd.read_csv(data_path+'misdiagnosed.csv')
display(misdiagnosed_df.head(5))
misdiagnosed_df['EPISODE_DATE'] = pd.to_datetime(misdiagnosed_df['EPISODE_DATE'])


# In[7]:


# Rename GLOBAL_PATIENT_DB_ID to patient_id
df_data_comp.rename(columns={"GLOBAL_PATIENT_DB_ID": "patient_id"}, inplace=True)


# In[8]:


# Merge With Misdiagnosed patients
df_data_comp= df_data_comp.merge(misdiagnosed_df[['patient_id', 'EPISODE_DATE', 'Misdiagnosed','Misdiagnosed1']], how='left', on=['patient_id', 'EPISODE_DATE'])
print (df_data_comp.shape)
display(df_data_comp.head(1))


# In[9]:


df_data_comp['Misdiagnosed1'] = df_data_comp['Misdiagnosed1'].replace(['NO', 'YES', 'UNKNOWN'],[0, 1, 2])
df_data_comp['Misdiagnosed'] = df_data_comp['Misdiagnosed'].replace(['NO', 'YES', 'UNKNOWN'],[0, 1, 2])


# In[10]:


for i, j in zip(df_data_comp, df_data_comp.dtypes):
    if not (j == "float64" or j == "int64" or j == 'uint8' or j == 'datetime64[ns]'):
        print(i, j)
        df_data_comp[i] = pd.to_numeric(df_data_comp[i], errors='coerce')

df_data_comp.shape


# In[11]:


df_data_comp = df_data_comp.replace([-1], [np.nan])


# In[12]:


df_data_comp = df_data_comp[df_data_comp['Misdiagnosed1']<2]
df_data_comp = df_data_comp.astype({col: str('float64') for col, dtype in zip (df_data_comp.columns.tolist(), df_data_comp.dtypes.tolist()) if 'int' in str(dtype) or str(dtype)=='object'})


# In[13]:


categorical_columns = [col for col in df_data_comp.columns if col in data_variable_cat.keys()]


# In[14]:


for column in categorical_columns:
    def replace_numerical_category(column, x):
        if x in data_variable_cat[column]:
            x = data_variable_cat[column][x]
        else:
            x = np.nan
        return x
    df_data_comp[column]=df_data_comp[column].apply(lambda x : replace_numerical_category(column, x))


# In[15]:


# replace with Unlnown
df_data_comp[categorical_columns] = df_data_comp[categorical_columns].replace([np.nan], ['Unknown'])
# df_data_comp[categorical_columns] = df_data_comp[categorical_columns].replace(['Not asked'], ['Unknown'])
# df_data_comp[categorical_columns] = df_data_comp[categorical_columns].replace(['Not known'], ['Unknown'])


# In[16]:


def find_mixed_type_list(l):
    for i in range(0,len(l)-1):
        if type(l[i])!=type(l[i+1]):
            return True
    return False
        
list_corrupted_columns = []        
for col in categorical_columns:
    if find_mixed_type_list(df_data_comp[col].unique().tolist()):
        list_corrupted_columns.append(col)
        print (col,': ',df_data_comp[col].unique().tolist())

print(len(list_corrupted_columns))


# In[17]:


for col in list_corrupted_columns:
    print (prepared_dataset.groupby(col)[col].count())


# In[18]:


df_data_comp[categorical_columns] = df_data_comp[categorical_columns].replace(['Unknown'], [np.nan])
df_data_comp.shape


# In[19]:


df_data_comp = df_data_comp.drop(columns=['patient_id', 'EPISODE_DATE', 'CAMDEX SCORES: MINI MENTAL SCORE', 'OPTIMA DIAGNOSES V 2010: PETERSEN MCI', 
                                          'Misdiagnosed', 'MINI MENTAL SCORE PRE', 'durations(years)', 'EPISODE'])


# In[20]:


df_data_comp_save = df_data_comp


# In[66]:


df_data_comp = df_data_comp_save


# In[67]:


# Take only columns which are filled for 133 misdiagnosed patients almost
df_data_comp_X_misdiag = df_data_comp[df_data_comp['Misdiagnosed1']==1]
df_data_comp_X_misdiag = drop_missing_columns(df_data_comp_X_misdiag[df_data_comp_X_misdiag.isna().sum(axis=1)<1400], 0.99) # thresold to decide about missing values 1506 in this case


df_data_comp = df_data_comp[df_data_comp_X_misdiag.columns]
df_data_comp.shape


# In[68]:


df_data_comp_save = df_data_comp


# In[69]:


df_data_comp = df_data_comp_save


# In[70]:


df_data_comp  = drop_missing_columns(df_data_comp[df_data_comp.isna().sum(axis=1)<5], 0.97)

print (df_data_comp[df_data_comp['Misdiagnosed1']==1].shape, df_data_comp[df_data_comp['Misdiagnosed1']==0].shape)


# In[71]:


# # feature transforamtion - one-hot encoding

prepared_dataset_exp = df_data_comp

# select categorical data columns

categorical_columns_final_exp = [col for col in prepared_dataset_exp.columns if col in categorical_columns]

new_prepared_data = prepared_dataset_exp.drop(categorical_columns_final_exp, axis=1)
for i in categorical_columns_final_exp:
    x = pd.get_dummies(prepared_dataset_exp[i]).add_prefix(i+'::')
    new_prepared_data = pd.concat([new_prepared_data, x], axis=1)

df_data_comp = new_prepared_data

df_data_comp.shape


# In[72]:


# drop Nagative Features # if there is only two values in columns only
# let it do later # for binary categroies

s1 = set([col.replace('::Incorrect', '') for col in df_data_comp.columns if 'Incorrect' in col.split('::')])-set([col.replace('::Correct', '') for col in df_data_comp.columns if 'Correct' in col.split('::')])
s2 = set([col.replace('::Yes', '') for col in df_data_comp.columns if 'Yes' in col.split('::')])-set([col.replace('::No', '') for col in df_data_comp.columns if 'No' in col.split('::')])
s3 = set([col.replace('::Correct', '') for col in df_data_comp.columns if 'Correct' in col.split('::')])-set([col.replace('::Incorrect', '') for col in df_data_comp.columns if 'Incorrect' in col.split('::')])
s4 = set([col.replace('::No', '') for col in df_data_comp.columns if 'No' in col.split('::')])-set([col.replace('::Yes', '') for col in df_data_comp.columns if 'Yes' in col.split('::')])

s = s1.union(s2).union(s3).union(s4)

s_list = list(s)

print (len(s_list))

# save df of s_list
exp_columns = [col for col in df_data_comp.columns if re.sub('::.*', '', col) in s_list and ('::No' in col or '::Incorrect' in col)]

print (exp_columns)
print (s_list)


# In[73]:


# drop Nagative Features # if there is only two values in columns only
df_data_comp = df_data_comp.drop(columns=[col for col in df_data_comp.columns if (('::Incorrect' in col or '::No' in col)) & (col not in exp_columns)])
print (df_data_comp.shape, df_data_comp.columns.tolist())


# In[74]:


print (df_data_comp.shape)
df_data_comp =  df_data_comp.dropna()
df_data_comp.shape


# In[75]:


# drop duplicates
df_data_comp.drop_duplicates(inplace=True)
df_data_comp.shape


# In[76]:


df_data_comp[df_data_comp['Misdiagnosed1']==0].shape, df_data_comp[df_data_comp['Misdiagnosed1']==1].shape


# In[77]:


# outlier detection
from sklearn.ensemble import IsolationForest
X = df_data_comp[df_data_comp['Misdiagnosed1']==0].drop(columns=['Misdiagnosed1'])
clf = IsolationForest(random_state=0).fit(X)
outlier_no_label = clf.predict(X)

from sklearn.ensemble import IsolationForest
X = df_data_comp[df_data_comp['Misdiagnosed1']==1].drop(columns=['Misdiagnosed1'])
clf = IsolationForest(random_state=0).fit(X)
outlier_yes_label = clf.predict(X)

print (sum(outlier_no_label)+ (len(outlier_no_label)-sum(outlier_no_label))/2)
print (sum(outlier_yes_label)+ (len(outlier_yes_label)-sum(outlier_yes_label))/2)


# In[78]:


df_data_comp['outlier_label'] = 0.0
df_data_comp.loc[df_data_comp['Misdiagnosed1']==0, 'outlier_label']=outlier_no_label
df_data_comp.loc[df_data_comp['Misdiagnosed1']==1, 'outlier_label']=outlier_yes_label
print (sum(df_data_comp['outlier_label']))


# In[79]:


sum(df_data_comp[df_data_comp['Misdiagnosed1']==0]['outlier_label']), sum(df_data_comp[df_data_comp['Misdiagnosed1']==1]['outlier_label'])


# In[80]:


df_X_y = df_data_comp[(df_data_comp['outlier_label']==1) | (df_data_comp['Misdiagnosed1']==1)]
df_X = df_X_y.drop(columns=['Misdiagnosed1'])
df_y = df_X_y['Misdiagnosed1']
print (df_X.shape, df_y.shape)


# In[81]:


X_full_imput, y_full_imput = df_X.values, df_y.values #X_full.values, y_full.values

# model training
rf_estimator = RandomForestClassifier(random_state=0)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=123)
important_features = set()
important_features_size = 40
for i, (train, test) in enumerate(cv.split(X_full_imput, y_full_imput)):
    rf_estimator.fit(X_full_imput[train], y_full_imput[train])
    y_predicted = rf_estimator.predict(X_full_imput[test])
    print (classification_report(y_full_imput[test], y_predicted))
    
    # print important features
    # model important feature
    fea_importance = rf_estimator.feature_importances_
    indices = np.argsort(fea_importance)[::-1]
    for f in range(important_features_size):
        # print("%d. feature: %s (%f)" % (f + 1, X_full.columns.values[indices[f]], fea_importance[indices[f]]))
        important_features.add(df_X.columns.values[indices[f]])
    #lime interpretability 
    '''explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_full_imput[train]), 
                                                       feature_names=[change_feature_names(fea) for fea in X_full.columns.values], 
                                                       class_names= ['No Dementia', 'Dementia'],#rf_estimator.classes_, 
                                                       discretize_continuous=True, random_state=123)
    exp = explainer.explain_instance(X_full_imput[test][5], rf_estimator.predict_proba, num_features=10)
    #exp.show_in_notebook(show_table=True, show_all=False)
    exp.save_to_file('model_1DT_'+str(i)+'.html')'''
    #print (exp.as_list())
    #fig = exp.as_pyplot_figure()
    #plt.show()
    
    # shap interpretability
    
#important feature list
print ('important_features: ', list(important_features))


# In[82]:


df_X, df_y = df_X[list(important_features)], df_y


# In[89]:


# Random Forest Classfier

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import classification_report
import graphviz
from sklearn import tree
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus, joblib
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from sklearn.model_selection import train_test_split
import re
from dtreeviz.trees import *

# patient_df_X_fill_data[patient_df_y_cat==0]
X, y = df_X, df_y
clf = RandomForestClassifier(n_estimators=100)
print (cross_validate(clf, X, y, scoring=['recall_macro', 'precision_macro', 'f1_macro', 'accuracy'], cv=5) )
# y_pred = cross_val_predict(clf,X, y, cv=5 )
# print(classification_report(y, y_pred, target_names=['NO','YES']))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

feature_names = df_X.columns

clf = tree.DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train) 
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print (classification_report(y_test, y_pred))

'''dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_names, 
               class_names=['NO', 'YES'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())'''

def change_feature_names(feature_name):
    feature_name = feature_name.replace('_',' ')
    p1 = '\w.*\d.*-\d.*:\s\(\d.*\w\)\s'
    p2 = '\w.*:\s'
    feature_name = re.sub(p1, '', feature_name)
    # feature_name = re.sub(p2, '', feature_name)
    for key, value in score_dict.items():
        if feature_name in key:
            feature_name = feature_name+'{}'.format(value)
    return feature_name

bool_feature_names_DT = df_X.select_dtypes(include='uint8').columns
feature_names_DT = [change_feature_names(i) for i in feature_names]
bool_feature_names_DT = [change_feature_names(i) for i in  bool_feature_names_DT] # Important 0: NO and 1: YES
bool_feature_names_true_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_1.0' in i ]
bool_feature_names_false_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_0.0' in i ]
feature_names_for_split_DT = [i for i in feature_names_DT if ' SCORE' in i] 


viz = dtreeviz(clf, 
               x_data=X_train,
               y_data=y_train,
               target_name='class',
               feature_names=feature_names_DT,
               bool_feature_names_true=bool_feature_names_true_DT,
               bool_feature_names_false=bool_feature_names_false_DT,
               feature_names_for_split=feature_names_for_split_DT,
               class_names=['misdiagnosed-No', 'misdiagnosed-Yes'],
               fancy=False, label_fontsize=40, ticks_fontsize=2)

viz.save('original_dataset.svg')
drawing = svg2rlg("./original_dataset.svg".format(i))
renderPDF.drawToFile(drawing, "./original_dataset.pdf".format(i))


# In[90]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto')
data_p_s, target_p_s = smote.fit_resample(df_X, df_y)
print (data_p_s.shape, target_p_s.shape)
# patient_df_X_fill_data[patient_df_y_cat==0]
X, y = data_p_s,  target_p_s
clf = RandomForestClassifier(n_estimators=100)
print (cross_validate(clf, X, y, scoring=['recall_macro', 'precision_macro', 'f1_macro', 'accuracy'], cv=5) )
# y_pred = cross_val_predict(clf,X, y, cv=5 )
# print(classification_report(y, y_pred, target_names=['NO','YES']))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

feature_names = df_X.columns

clf = tree.DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train) 
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print (classification_report(y_test, y_pred))

'''dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_names, 
               class_names=['NO', 'YES'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())'''

bool_feature_names_DT = df_X.select_dtypes(include='uint8').columns
feature_names_DT = [change_feature_names(i) for i in feature_names]
bool_feature_names_DT = [change_feature_names(i) for i in  bool_feature_names_DT] # Important 0: NO and 1: YES
bool_feature_names_true_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_1.0' in i ]
bool_feature_names_false_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_0.0' in i ]
feature_names_for_split_DT = [i for i in feature_names_DT if ' SCORE' in i] 


viz = dtreeviz(clf, 
               x_data=X_train,
               y_data=y_train,
               target_name='class',
               feature_names=feature_names_DT,
               bool_feature_names_true=bool_feature_names_true_DT,
               bool_feature_names_false=bool_feature_names_false_DT,
               feature_names_for_split=feature_names_for_split_DT,
               class_names=['misdiagnosed-No', 'misdiagnosed-Yes'],
               fancy=False, label_fontsize=40, ticks_fontsize=2)

viz.save('oversampled_smote.svg')
drawing = svg2rlg("./oversampled_smote.svg".format(i))
renderPDF.drawToFile(drawing, "./oversampled_smote.pdf".format(i))


# In[91]:


from collections import Counter
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(df_X, df_y)
print(sorted(Counter(y_resampled).items()))
X, y = X_resampled,  y_resampled
clf = RandomForestClassifier(n_estimators=100)
print (cross_validate(clf, X, y, scoring=['recall_macro', 'precision_macro', 'f1_macro', 'accuracy'], cv=5) )
# y_pred = cross_val_predict(clf,X, y, cv=5 )
# print(classification_report(y, y_pred, target_names=['NO','YES']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

feature_names = df_X.columns

clf = tree.DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train) 
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print (classification_report(y_test, y_pred))

'''dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_names, 
               class_names=['NO', 'YES'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())'''


bool_feature_names_DT = df_X.select_dtypes(include='uint8').columns
feature_names_DT = [change_feature_names(i) for i in feature_names]
bool_feature_names_DT = [change_feature_names(i) for i in  bool_feature_names_DT] # Important 0: NO and 1: YES
bool_feature_names_true_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_1.0' in i ]
bool_feature_names_false_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_0.0' in i ]
feature_names_for_split_DT = [i for i in feature_names_DT if ' SCORE' in i] 


viz = dtreeviz(clf, 
               x_data=X_train,
               y_data=y_train,
               target_name='class',
               feature_names=feature_names_DT,
               bool_feature_names_true=bool_feature_names_true_DT,
               bool_feature_names_false=bool_feature_names_false_DT,
               feature_names_for_split=feature_names_for_split_DT,
               class_names=['misdiagnosed-No', 'misdiagnosed-Yes'],
               fancy=False, label_fontsize=40, ticks_fontsize=2)

viz.save('undersampled_clustercentroid.svg')
drawing = svg2rlg("./undersampled_clustercentroid.svg".format(i))
renderPDF.drawToFile(drawing, "./undersampled_clustercentroid.pdf".format(i))


# In[92]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X, y = rus.fit_resample(df_X, df_y)
clf = RandomForestClassifier(n_estimators=100)
print (cross_validate(clf, X, y, scoring=['recall_macro', 'precision_macro', 'f1_macro', 'accuracy'], cv=5) )
# y_pred = cross_val_predict(clf,X, y, cv=5 )
# print(classification_report(y, y_pred, target_names=['NO','YES']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

feature_names = df_X.columns

clf = tree.DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_train, y_train) 
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print (classification_report(y_test, y_pred))

'''dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_names, 
               class_names=['NO', 'YES'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())'''


bool_feature_names_DT = df_X.select_dtypes(include='uint8').columns
feature_names_DT = [change_feature_names(i) for i in feature_names]
bool_feature_names_DT = [change_feature_names(i) for i in  bool_feature_names_DT] # Important 0: NO and 1: YES
bool_feature_names_true_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_1.0' in i ]
bool_feature_names_false_DT = [i for i in bool_feature_names_DT if '::' in i] #('IDENTIFIES' in i or 'RECALL' in i) and '_0.0' in i ]
feature_names_for_split_DT = [i for i in feature_names_DT if ' SCORE' in i] 


viz = dtreeviz(clf, 
               x_data=X_train,
               y_data=y_train,
               target_name='class',
               feature_names=feature_names_DT,
               bool_feature_names_true=bool_feature_names_true_DT,
               bool_feature_names_false=bool_feature_names_false_DT,
               feature_names_for_split=feature_names_for_split_DT,
               class_names=['misdiagnosed-No', 'misdiagnosed-Yes'],
               fancy=False, label_fontsize=40, ticks_fontsize=2)

viz.save('undersampled_random.svg')
drawing = svg2rlg("./undersampled_random.svg".format(i))
renderPDF.drawToFile(drawing, "./undersampled_random.pdf".format(i))

