#!/usr/bin/env python
# coding: utf-8

# ------
# **Dementia Patients -- Analysis and Prediction**
### ***Author : Akhilesh Vyas***
### ****Date : Januaray, 2020****

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import re


# In[2]:


# define path
data_path = '../../../datalcdem/data/optima/dementia_18July/class_fast_normal_slow_api_inputs/'
result_path = '../../../datalcdem/data/optima/dementia_18July/class_fast_normal_slow_api_inputs/results/'

# define
best_params_rf_dict = {'n_estimators': 25, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 8, 'bootstrap': False}
class_names_dict = {'Slow':0, 'Slow_MiS':1, 'Normal':2, 'Normal_MiS':3, 'Fast':4, 'Fast_MiS':5}


# In[3]:


def replaceNamesComandMed(col):
    #col1 = re.sub(r'[aA-zZ]*_cui_', 'Episode', col)
    if 'Medication_cui_' in col:
        return 'Medication_cui_'+str('0_')+col.split('_')[-1]
    if 'Comorbidity_cui_' in col:
        return 'Comorbidity_cui_'+str('0_')+col.split('_')[-1]
    if 'Age_At_Episode' in col:
        # print ('Age_At_Episode_'+str('0'))
        return 'Age_At_Episode_'+str('0')
    return col

def replacewithEpisode(col):
    if 'Medication_cui_' in col:
        return 'Episode'+str(col.split('_')[-2])+'_Med_'+col.split('_')[-1]
    if 'Comorbidity_cui_' in col:
        return 'Episode'+str(col.split('_')[-2])+'_Com_'+col.split('_')[-1]
    if 'Age_At_Episode' in col:
        return 'Episode'+str(col.split('_')[-1])+'_Age'
    return col


# In[4]:


# read dataframes
df_fea_all = pd.read_csv(data_path+'final_features_file_without_feature_selection_smote.csv')
# print (df_fea_all.shape)
df_fea_rfecv = pd.read_csv(data_path+'final_features_file_with_feature_selection_rfecv.csv')
df_fea_rfecv.rename(columns={col:col.replace('_TFV_', '_') for col in df_fea_rfecv.columns}, inplace=True)
df_fea_rfecv.rename(columns={col:replaceNamesComandMed(col) for col in df_fea_rfecv.columns.tolist() if not bool(re.search(r'_[1-9]_?',col))}, inplace=True)
df_fea_rfecv.rename(columns={col:replacewithEpisode(col) for col in df_fea_rfecv.columns.tolist()}, inplace=True)
df_fea_rfecv.rename(columns={'CAMDEX SCORES: MINI MENTAL SCORE_CATEGORY_Mild':'Initial_MMSE_Score_Mild', 
                             'CAMDEX SCORES: MINI MENTAL SCORE_CATEGORY_Moderate':'Initial_MMSE_Score_Moderate'}, inplace=True)
# print(df_fea_rfecv.shape)

# read object
data_p_i = pickle.load(open(data_path + 'data_p_i.pickle', 'rb'))
target_p_i = pickle.load(open(data_path + 'target_p_i.pickle', 'rb')) 
rfecv_support_ = pickle.load(open(data_path + 'rfecv.support_.pickle', 'rb'))
# print(data_p_i.shape, target_p_i.shape, rfecv_support_.shape)

#read dictionary
# Treatment data
treatmnt_df = pd.read_csv(data_path+'Treatments.csv')
# print(treatmnt_df.head(5))
treatmnt_dict = dict(zip(treatmnt_df['name'], treatmnt_df['CUI_ID']))
# print ('\n Unique Treatment data size: {}\n'.format(len(treatmnt_dict)))

# Comorbidities data
comorb_df = pd.read_csv(data_path+'comorbidities.csv')
# print(comorb_df.head(5))
comorb_dict = dict(zip(comorb_df['name'], comorb_df['CUI_ID']))
# print ('\n Unique Comorbidities data size: {}\n'.format(len(comorb_dict)))


# In[5]:
# for i, j in zip(df_fea_rfecv.columns.to_list(), pd.read_csv(data_path+'final_features_file_with_feature_selection_rfecv.csv').columns.tolist()):
#    print (j,'   ', i)
# In[6]:


# Classification Model
data_p_grid = data_p_i[:,rfecv_support_]
rf_bp = best_params_rf_dict

rf_classifier=RandomForestClassifier(n_estimators=rf_bp["n_estimators"],
                                     min_samples_split=rf_bp['min_samples_split'],
                                     min_samples_leaf=rf_bp['min_samples_leaf'],
                                     max_features=rf_bp['max_features'],
                                     max_depth=rf_bp['max_depth'],
                                     bootstrap=rf_bp['bootstrap'])
rf_classifier.fit(data_p_grid, target_p_i)


# Plot Tree
# plot randomForest
'''estimator_id = 1  #need to change
estimator = rf_classifier.estimators_[1]

feature_names = # extract from df_fea_rfecv

export_graphviz(estimator, out_file=result_path+'tree.dot', 
                feature_names = feature_names,
                class_names = class_names_p,  # Extract from class_names_dict 
                rounded = True, proportion = False, 
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', result_path+'tree.dot', '-o', result_path+'tree.png', '-Gdpi=600'])
call(['dot', '-Tpdf', result_path+'tree.dot', '-o', result_path+'tree.pdf', '-Gdpi=600']) '''

# Example for testing classification model
'''st_ix_p = 21
end_ix_p = 22
p = data_p_grid[st_ix_p:end_ix_p]
t = target_p_i[st_ix_p:end_ix_p]
print ('Mean Accuracy: ', rf_classifier.score(data_p_grid, target_p_i)*100)
print ('Predict Probability: ', rf_classifier.predict_proba(p)*100)
print ('Prediction: ', rf_classifier.predict(p))
print ('Target: ', t)'''


# In[48]:
# patient_data_in = {'gender':['Female'], 'dementia':['True'], 'smoker':['no_smoker'], 'alcohol':'mild_drinking', 'education':['medium'], 'bmi':23, 'weight':60,
#  'apoe':['E3E3'], 'Initial_MMSE_Score':['Mild'], 'Episode1_Com':['C0002965', 'C0042847'], 'Episode1_Med':['C0014695', 'C0039943'], 'Episode1_Age':67, 'Episode2_Com':['C0032533'],  # 'Episode2_Med':['C1166521'], 'Episode2_Age':69}


def create_patient_feature_vector(patient_data_in):
    print ('create_patient_feature_vector')
    patient_data = pd.DataFrame(data=np.zeros(shape=df_fea_rfecv.iloc[0:1, 0:-1].shape), columns=df_fea_rfecv.columns[0:-1])
    print (patient_data.shape)
    for key,value in patient_data_in.items():
        try:
            if type(value)==list:
                if len(value)>0:
                    print (key, value)
                    for i in value:
                        if key+'_'+str(i) in patient_data.columns.tolist():
                            print ('##############', key+'_'+str(i))
                            patient_data.at[0, key+'_'+str(i)] = 1.0                 
            elif key+'_'+str(value) in patient_data.columns.tolist():
                print ('#########', key+'_'+str(value))
                patient_data.at[0, key+'_'+str(value)] = 1.0
            elif key in patient_data.columns.tolist():
                print ('########',key)
                patient_data.at[0, key] = value
        except Exception as e:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print (message)
            return None
    
    return patient_data


def result(patient_vect):
    print (patient_vect.shape)
    print (patient_vect)
    print ('result')
    try:
        print ('Predict Probability: ', rf_classifier.predict_proba(patient_vect)*100)
        print ('Prediction: ', rf_classifier.predict(patient_vect))
        predicted_class = list(class_names_dict.keys())[rf_classifier.predict(patient_vect)[0]]
        class_probability = {i:j for i, j in zip(list(class_names_dict.keys()), rf_classifier.predict_proba(patient_vect)[0])}
        response= { 'modelName': 'Classification of Dementia Patient Progression',
                'class_probability': class_probability,
                'predicted_class': predicted_class}
        return response
    
    except Exception as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print (message)
        return None


def main():
    # patient_data_in to be get by Web API
    try:
        print ('Shape of patient vector with all selected Feature', df_fea_all.shape)
        print('Shape of patient vector with selected Feature', df_fea_rfecv.shape)
        print(data_p_i.shape, target_p_i.shape, rfecv_support_.shape)
        print('Medication:\n', treatmnt_df.head(5))
        print ('\n Unique Treatment data size: {}\n'.format(len(treatmnt_dict)))
        print('Comorbidities\n', comorb_df.head(5))
        print ('\n Unique Comorbidities data size: {}\n'.format(len(comorb_dict)))
        pat_df = create_patient_feature_vector(patient_data_in)
        print ('pat_df', pat_df)
        patient_data_fea = pat_df.values.reshape(1,-1)
        response = result(patient_data_fea)
        print (response)
        return response
    except Exception as e:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print (message)
        return None
  

print("\n\n######## Predictive Model for Dementia Patients#############\n\n")
main()
