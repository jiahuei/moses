# -*- coding: utf-8 -*-
"""
Created on 14 Apr 2020 01:00:52

@author: jiahuei
"""
import os
import pandas as pd

pjoin = os.path.join

data_root = pjoin('/mol_data', 'DeepAffinity')
pair_files = ['EC50_protein_compound_pair.tsv',
              'IC50_protein_compound_pair.tsv',
              'Kd_protein_compound_pair.tsv',
              'Ki_protein_compound_pair.tsv', ]
mapping_files = ['dpid_seq.tsv', 'dcid_smi.tsv']

id_cols = ['DeepAffinity Protein ID', 'DeepAffinity Compound ID']
df_final = None
for tsv in pair_files:
    tsv_full_path = pjoin(data_root, tsv)
    df = pd.read_csv(tsv_full_path, delimiter='\t')
    cols = df.columns
    df = df.loc[:, id_cols + [cols[-1]]]
    if df_final is None:
        df_final = df
    else:
        df_final = df_final.merge(right=df, how='outer', on=id_cols)
    print('{} length: {}'.format(tsv, len(df)))
print('Final merged DF length: {}'.format(len(df_final)))

df_final = df_final.fillna(-1.)
print(df_final)

# Merge with SMILES and Protein seq
for i, tsv in enumerate(mapping_files):
    df = pd.read_csv(pjoin(data_root, tsv), delimiter='\t')
    df_final = df_final.merge(right=df, how='inner', on=id_cols[i])
print(df_final)

with open(pjoin(data_root, 'merged_data.tsv'), 'w') as f:
    df_final.to_csv(f, sep='\t', line_terminator='\n')
