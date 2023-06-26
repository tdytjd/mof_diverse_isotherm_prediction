#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:03:52 2023

@author: yuxiaohan
"""

#calculate GCMC and ML binary with IAST 

import pandas as pd
from binary_calculation import generate_IAST_predictions_GCMC
import sys

data_set_3=pd.read_csv('Set3_with_predictions.csv')

mof_list=list(data_set_3.MOF.unique())

#user defined index
pair_num_1 = int(sys.argv[1])
pair_num_2 = int(sys.argv[2])


mol_list=['3-butenal','butylamine','tert-butanol','4-methyl-1-hexene','4,4-dimethyl-1-pentene','2,2-dimethylpentane',\
'2,4-dimethylpentane','3,3-dimethylpentane','methylethylpropylamine','dimethylbutylamine','ethyl tert-butyl ether','diisopropyl ether']

mol_Pvp_list=[9340.2,12407.9,7622.54,11546.5,16063.9,14383.2,11743.4,14383.2,10257.4,10135.2,17013.5,20795.4]
    

data_mol_1=data_set_3[data_set_3['molecule']==mol_list[pair_num_1]]
data_mol_2=data_set_3[data_set_3['molecule']==mol_list[pair_num_2]]

P_IAST=0.5*(mol_Pvp_list[pair_num_1]+mol_Pvp_list[pair_num_2])

print(mol_list[pair_num_1],mol_list[pair_num_2],P_IAST)
result_dataframe=generate_IAST_predictions_GCMC(data_mol_1,data_mol_2,P_IAST,mof_list)
result_dataframe.to_csv('set_3_pair_'+str(pair_num_1)+str(pair_num_2)+'/'+mol_list[pair_num_1]+mol_list[pair_num_2]+'.csv',index=False)
