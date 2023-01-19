import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Daten einlesen
alzheimer_data = pd.read_csv("C:/Users/KDick/PycharmProjects/pythonProject1/inputs/Alzheimer/oasis_longitudinal.csv")

print('')
print('Das sind die eingelesenen Daten:\n', alzheimer_data.head())

# Die für den Aufbau nicht benötigten Spalten Subject ID MRI ID Group und Hand aus den Daten löschen.
# Hand wurde herausgenommen, da nur rechtshänder in der Versuchsgruppe vertreten waren und die information somit
# irrelevant zur Auswertung der Daten wurde.
# Subject ID und MR ID wurden herausgenommen, da alle 373 Daten einen individuellen Wert in diesen beiden Spalten
# aufweisen. Daher sind diese Werte auch irrelevant zur Auswertung.

alzheimer_data_clean = alzheimer_data.drop(['Subject ID', 'MRI ID', 'Group', 'Hand'], axis=1)
print('')
print('Das sind die eíngelesenen Daten ohne die Spalten Subject ID, MRI ID, Group und Hand:\n',
      alzheimer_data_clean.head())

print('')
print('Hier sieht man in welchen Spalten wie viele Werte fehlen:\n', alzheimer_data_clean.isna().sum())

print('Hier die Beschreibung der Werteverteilung von SES (1.) und MMSE (2.): \n', alzheimer_data_clean['SES'].describe(), '\n',
      alzheimer_data_clean['MMSE'].describe())

alzheimer_data_clean['SES'].fillna((alzheimer_data_clean['SES'].mean()), inplace=True)
alzheimer_data_clean['MMSE'].fillna((alzheimer_data_clean['MMSE'].mean()), inplace=True)

print('')
print("Das sind die Daten ohne NaN:\n", alzheimer_data_clean.head())

alzheimer_data_clean.rename(columns={'MR Delay': 'mrdelay', 'Visit': 'visit', 'Age': 'age', 'EDUC': 'educ',
                                     'SES': 'ses', 'MMSE': 'mmse', 'CDR': 'cdr', 'eTIV': 'etiv', 'nWBV': 'nwbv',
                                     'ASF': 'asf', 'M/F': 'm_f'}, inplace=True)
print('')
print('Hier sind die Daten nocheinmal mit den umbenannten Spalten: \n', alzheimer_data_clean.head())

alzheimer_data_temp = alzheimer_data_clean.drop(['m_f'], axis=1)

cdr = alzheimer_data_temp['cdr']
alzheimer_data_covariance = alzheimer_data_temp.cov()
print('')
print('Hier sind die Kovarianzergebnisse der Daten:\n', alzheimer_data_covariance)

alzheimer_data_correlation = alzheimer_data_temp.corr()
print('')
print('Hier sind die Korrelationskoeffizienten der Daten:\n', alzheimer_data_correlation)
print('')
print('Hier die CDR Spalte:\n', alzheimer_data_correlation['cdr'])

nul = 0
nulfive = 0
one = 0
two = 0

for value in alzheimer_data_temp['cdr']:
    if value == 0.0:
        nul = nul +1
    if value == 0.5:
        nulfive = nulfive + 1
    if value == 1.0:
        one = one + 1
    if value == 2.0:
        two = two + 1

print('')
print('Anzahl der CDR = 0: ', nul, '\n Anzahl der CDR = 0.5: ', nulfive, '\n Anzahl der CDR = 1: ', one,
      '\n Anzahl der CDR = 2: ', two)

visit_corr = pearsonr(alzheimer_data_temp['visit'], cdr)
mrdelay_corr = pearsonr(alzheimer_data_temp['mrdelay'], cdr)
age_corr = pearsonr(alzheimer_data_temp['age'], cdr)
educ_corr = pearsonr(alzheimer_data_temp['educ'], cdr)
ses_corr = pearsonr(alzheimer_data_temp['ses'], cdr)
mmse_corr = pearsonr(alzheimer_data_temp['mmse'], cdr)
etiv_corr = pearsonr(alzheimer_data_temp['etiv'], cdr)
nwbv_corr = pearsonr(alzheimer_data_temp['nwbv'], cdr)
asf_corr = pearsonr(alzheimer_data_temp['asf'], cdr)

asf_etiv = pearsonr(alzheimer_data_temp['asf'], alzheimer_data_temp['etiv'])
ses_educ = pearsonr(alzheimer_data_temp['ses'], alzheimer_data_temp['educ'])
nwbv_age = pearsonr(alzheimer_data_temp['nwbv'], alzheimer_data_temp['age'])
mrdelay_visit = pearsonr(alzheimer_data_temp['mrdelay'], alzheimer_data_temp['visit'])
nwbv_etiv = pearsonr(alzheimer_data_temp['nwbv'], alzheimer_data_temp['etiv'])
nwbv_mmse = pearsonr(alzheimer_data_temp['nwbv'], alzheimer_data_temp['mmse'])
etiv_educ = pearsonr(alzheimer_data_temp['etiv'], alzheimer_data_temp['educ'])
etiv_ses = pearsonr(alzheimer_data_temp['etiv'], alzheimer_data_temp['ses'])
nwbv_asf = pearsonr(alzheimer_data_temp['nwbv'], alzheimer_data_temp['asf'])
age_mrdelay = pearsonr(alzheimer_data_temp['age'], alzheimer_data_temp['mrdelay'])

print('')
print('Hier sind sowohl die Korrelationsdaten, als auch deren statistische Signifikanzwerte zur CDR Spalte:\n',
      'Visit: ', visit_corr, '\n','MR Delay: ', mrdelay_corr, '\n', 'Age: ', age_corr, '\n', 'EDUC: ', educ_corr, '\n',
      'SES: ', ses_corr, '\n', 'MMSE: ', mmse_corr, '\n', 'eTIV: ', etiv_corr, '\n', 'nWBV: ', nwbv_corr, '\n',
      'ASF: ', asf_corr)
print('')
print('Hier die Signifikanzwerte der Korrelationswerte über |+/- 0.2| nicht direkt CDR betreffend:', '\n ASF_ETIV: ', asf_etiv,
      '\n SES_EDUC: ', ses_educ, '\n nWBV_Age: ', nwbv_age, '\n MR_Delay_Visit: ', mrdelay_visit, '\n nWBV_eTIV: ',
      nwbv_etiv, '\n nWBV_MMSE: ', nwbv_mmse, '\n eTIV_EDUC: ', etiv_educ, '\n eTIV_SES: ', etiv_ses, '\n nWBV_ASF: ',
      nwbv_asf, '\n Age_MR_Delay: ', age_mrdelay)

plt.figure(figsize=(14,8))
corr_heat = sns.heatmap(alzheimer_data_correlation, annot=True)
plt.show()
