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

data_set_1=pd.read_csv('Set1_with_predictions.csv')
data_set_2=pd.read_csv('Set2_with_predictions.csv')

mof_list=list(data_set_1.MOF.unique())

#user defined index
pair_num = int(sys.argv[1])

mol_1_list=['C3H5N_propionitrile','C4H10O_methyl-isopropyl-ether','C3H8O_propylalcohol',\
            'C2H3N_acetonitrile','C7H14_4-methyl-1-hexene','C2H4O_acetaldehyde',\
                'C2H4O_acetaldehyde','C4H10O_methyl-propyl-ether','C3H6_propene',\
                    'C2H7N_ethylamine','C3H9N_propylamine','C6H10_1,5-hexadiene','C5H12O_3-methyl-1-butanol']
mol_2_list=['C7H12_1,5-heptadiene','C5H12_neopentane','C5H10O_methylpropylketone',\
            'C3H8O_isopropylalcohol','C7H14_4,4-dimethyl-1-pentene','C2H7N_ethylamine',\
                'C2H7N_dimethylamine','C5H10_2-pentene','C3H8_propane',\
                    'C2H7N_dimethylamine','C3H6O_propionaldehyde','C6H12_2-hexene','C5H10O_1-methyl-3-buten-1-ol']
P_IAST_list=[(6287.180000+6.997012e+03)/2,(91684.924170+1.707030e+05)/2,(3878.400000+5180.290000)/2,\
        (11153.500000+11634.900000)/2,(1.154653e+04+1.606392e+04)/2,(1.160790e+05+1.436356e+05)/2,\
            (1.160790e+05+1.933020e+05)/2,(73498.965200+76304.176270)/2,(1.560410e+06+1.023210e+06)/2,\
                (1.436356e+05+1.933020e+05)/2,(42066.518200+44312.056200)/2,(28726.473300+2.246805e+04)/2,\
                    (6.301872e+02+880.822110)/2]
    
if pair_num<10:
    data_mol_1=data_set_1[data_set_1['molecule']==mol_1_list[pair_num]]
    data_mol_2=data_set_1[data_set_1['molecule']==mol_2_list[pair_num]]
else:
    data_mol_1=data_set_2[data_set_2['molecule']==mol_1_list[pair_num]]
    data_mol_2=data_set_2[data_set_2['molecule']==mol_2_list[pair_num]]

P_IAST=P_IAST_list[pair_num]

print(mol_1_list[pair_num],mol_2_list[pair_num],P_IAST)
result_dataframe=generate_IAST_predictions_GCMC(data_mol_1,data_mol_2,P_IAST,mof_list)
result_dataframe.to_csv('pair_'+str(pair_num)+'/'+mol_1_list[pair_num]+mol_2_list[pair_num]+'.csv',index=False)
