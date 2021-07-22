import plotly.figure_factory as ff
import plotly.offline as py

# If an increase of >= 9 MMSE points occurs within the first year after a diagnosis of Alz or MCI OR if an increase of >=4 occurs at any time thereafter.
misdiag_pat = patient_com_treat_fea_raw_df[patient_com_treat_fea_raw_df['Misdiagnosed']=='YES']['patient_id'].unique()

patient_dur_mci_id = patient_com_treat_fea_raw_df[patient_com_treat_fea_raw_df['patient_id'].isin(misdiag_pat)][['patient_id', 'Misdiagnosed', 'durations(years)']].dropna()
patient_cat_dur_id_pivot = patient_dur_mci_id.pivot(index='patient_id', columns='durations(years)', values='Misdiagnosed')
patient_cat_dur_id_pivot.replace(['NO', 'YES'],[0,100],inplace=True)
patient_cat_dur_id_pivot.interpolate(method='linear', axis=1, limit_area='inside',  inplace=True)
display(patient_cat_dur_id_pivot.head(5))
patient_ids = ["P_ID:"+str(i) for i in patient_cat_dur_id_pivot.index.values]
line_patients = patient_cat_dur_id_pivot.columns.values
mms_values = patient_cat_dur_id_pivot.values


patient_dur_mci_id = patient_com_treat_fea_raw_df[patient_com_treat_fea_raw_df['patient_id'].isin(misdiag_pat)][['patient_id', 'MINI_MENTAL_SCORE', 'durations(years)']].dropna()
patient_cat_dur_id_pivot = patient_dur_mci_id.pivot(index='patient_id', columns='durations(years)', values='MINI_MENTAL_SCORE')
patient_cat_dur_id_pivot.interpolate(method='linear', axis=1, limit_area='inside',  inplace=True)
display(patient_cat_dur_id_pivot.head(5))
patient_ids_1 = ["P_ID:"+str(i) for i in patient_cat_dur_id_pivot.index.values]
line_patients_1 = patient_cat_dur_id_pivot.columns.values
mms_values_1 = patient_cat_dur_id_pivot.values

patient_dur_mci_id = patient_com_treat_fea_raw_df[patient_com_treat_fea_raw_df['patient_id'].isin(misdiag_pat)][['patient_id', 'PETERSEN_MCI', 'durations(years)']].dropna()
patient_cat_dur_id_pivot = patient_dur_mci_id.pivot(index='patient_id', columns='durations(years)', values='PETERSEN_MCI')
patient_cat_dur_id_pivot.replace([9.0],[np.nan],inplace=True)
patient_cat_dur_id_pivot.interpolate(method='linear', axis=1, limit_area='inside',  inplace=True)
display(patient_cat_dur_id_pivot.head(5))
patient_ids_2 = ["P_ID:"+str(i) for i in patient_cat_dur_id_pivot.index.values]
line_patients_2 = patient_cat_dur_id_pivot.columns.values
mms_values_2 = patient_cat_dur_id_pivot.values

fig = make_subplots(rows=1, cols=3)

fig.add_trace(go.Heatmap(
        z=mms_values,
        x=line_patients,
        y=patient_ids,
        colorscale='Viridis', 
        colorbar={"len":0.3, "y":0.1, "title":"Misdiagnosed: NO(0), YES(100)", 'titleside':'bottom'},
        showscale=True),
        row=1, col=1)

fig.add_trace(go.Heatmap(
        z=mms_values_1,
        x=line_patients_1,
        y=patient_ids_1,
        colorscale='Inferno', 
        colorbar={"len":0.5, "y":0.5, "title":"MINI_MENTAL_SCORE", 'titleside':'bottom'},
        showscale=True),
        row=1, col=2)

fig.add_trace(go.Heatmap(
        z=mms_values_2,
        x=line_patients_2,
        y=patient_ids_2,
        colorscale='Bluered_r', 
        colorbar={"len":0.3, "y":0.9, "title":"PETERSEN_MCI", 'titleside':'bottom'},
        showscale=True),
        row=1, col=3)
              
fig.update_layout(width=1500, height=500, title='Patient misdiagnosed during years of treatment')

#fig.show()

py.plot(fig,filename='plot_new_misdiagnos_duration_patientid_heatmap.html')