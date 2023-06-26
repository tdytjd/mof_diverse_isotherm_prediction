import pyiast
import numpy as np
import pandas as pd

from fitting_helper import fit_single_isotherms_iast


def generate_IAST_predictions_GCMC(data_mol_1,data_mol_2,P_IAST,mof_list):
    #data_mol_1,data_set_2 datasets of the two mols respectively with ML predictions and GCMC loadings and errors
    #mol_1, mol_2 names of the two molecules
    #P_IAST pressure of binary, equimolar of all cases
    #mof_list 350 mofs in the dataset

    column_GCMC=['Loading1_GCMC','Loading2_GCMC','Selectivity_GCMC']
    column_ML=['Loading1_ML','Loading2_ML','Selectivity_ML']

    pyiast_results=pd.DataFrame(columns=['MOFs','Reasonable','No_Adsorption_1','No_Adsorption_2','Cat_predicted_1','Cat_predicted_2']+column_GCMC+column_ML,index=range(len(mof_list)),dtype='object')

    for i in range(len(mof_list)):#
        print(i)
        pyiast_results.iloc[i,0]=mof_list[i]
        data_1=data_mol_1[data_mol_1['MOF']==mof_list[i]]
        data_2=data_mol_2[data_mol_2['MOF']==mof_list[i]]

        #what happens if one or both of the loading simulations are not reasonable?
        #direct errors for all
        if data_1.empty or data_2.empty:
            pyiast_results.iloc[i,1]='No'
        else:
            pyiast_results.iloc[i,1]='Yes'

            pyiast_results.iloc[i,2]=data_1['CatPredicted'].iloc[0]
            pyiast_results.iloc[i,3]=data_2['CatPredicted'].iloc[0]
            pyiast_results.iloc[i,4]=data_1['No Adsorption'].iloc[0]
            pyiast_results.iloc[i,5]=data_2['No Adsorption'].iloc[0]

            #make a list to store the fitted isotherms
            isotherm_1_GCMC_list=[]
            isotherm_2_GCMC_list=[]
            isotherm_1_ML_list=[]
            isotherm_2_ML_list=[]
            for k in range(10):
                isotherm_1_GCMC_list.append('I1_GCMC_'+str(k))
                isotherm_2_GCMC_list.append('I2_GCMC_'+str(k))
                isotherm_1_ML_list.append('I1_ML_'+str(k))
                isotherm_2_ML_list.append('I2_ML_'+str(k))

            L_GCMC_1=[]
            L_GCMC_2=[]
            S_GCMC=[]

            L_ML_1=[]
            L_ML_2=[]
            S_ML=[]

            #make sure the keys are 'pressure (Pa)' and 'loading (mol/kg)'
            #generate 100 pyIAST results for GCMC and ML respectively
            for j in range(10):
                #do the fitting process for data1
                df_mol_1_GCMC=pd.DataFrame(columns=['pressure (Pa)','loading (mol/kg)'],index=range(data_1.shape[0]),dtype='object')
                df_mol_1_ML=pd.DataFrame(columns=['pressure (Pa)','loading (mol/kg)'],index=range(data_1.shape[0]),dtype='object')

                df_mol_1_GCMC.iloc[:,0]=data_1['pressure'].values
                df_mol_1_ML.iloc[:,0]=data_1['pressure'].values

                df_mol_1_GCMC.iloc[:,1]=data_1.iloc[:,j-20].values
                df_mol_1_ML.iloc[:,1]=data_1.iloc[:,j-10].values

                if max(data_1.iloc[:,j-20].values)!=0:
                    K_guess_1_GCMC=data_1['K'].values[0]/max(data_1.iloc[:,j-20].values)
                else:
                    K_guess_1_GCMC=data_1['K'].values[0]

                if max(data_1.iloc[:,j-10].values)!=0:
                    K_guess_1_ML=data_1['K'].values[0]/max(data_1.iloc[:,j-10].values)
                else:
                    K_guess_1_ML=data_1['K'].values[0]

                isotherm_1_GCMC=fit_single_isotherms_iast(df_mol_1_GCMC,max(data_1['pressure'].values),K_guess_1_GCMC)
                isotherm_1_ML=fit_single_isotherms_iast(df_mol_1_ML,max(data_1['pressure'].values),K_guess_1_ML)

                globals()[isotherm_1_GCMC_list[j]]=isotherm_1_GCMC
                globals()[isotherm_1_ML_list[j]]=isotherm_1_ML

                #do the fitting process for data2
                df_mol_2_GCMC=pd.DataFrame(columns=['pressure (Pa)','loading (mol/kg)'],index=range(data_2.shape[0]),dtype='object')
                df_mol_2_ML=pd.DataFrame(columns=['pressure (Pa)','loading (mol/kg)'],index=range(data_2.shape[0]),dtype='object')

                df_mol_2_GCMC.iloc[:,0]=data_2['pressure'].values
                df_mol_2_ML.iloc[:,0]=data_2['pressure'].values

                df_mol_2_GCMC.iloc[:,1]=data_2.iloc[:,j-20].values
                df_mol_2_ML.iloc[:,1]=data_2.iloc[:,j-10].values

                if max(data_2.iloc[:,j-20].values)!=0:
                    K_guess_2_GCMC=data_2['K'].values[0]/max(data_2.iloc[:,j-20].values)
                else:
                    K_guess_2_GCMC=data_2['K'].values[0]

                if max(data_2.iloc[:,j-10].values)!=0:
                    K_guess_2_ML=data_2['K'].values[0]/max(data_2.iloc[:,j-10].values)
                else:
                    K_guess_2_ML=data_2['K'].values[0]

                isotherm_2_GCMC=fit_single_isotherms_iast(df_mol_2_GCMC,max(data_2['pressure'].values),K_guess_2_GCMC)
                isotherm_2_ML=fit_single_isotherms_iast(df_mol_2_ML,max(data_2['pressure'].values),K_guess_2_ML)

                globals()[isotherm_2_GCMC_list[j]]=isotherm_2_GCMC
                globals()[isotherm_2_ML_list[j]]=isotherm_2_ML

            #make all combinations of 10*10
            #calculate L and S

            for a in range(10):
                for b in range(10):
                    for fraction_guess in ([0.5,0.5],[0.9,0.1],[0.1,0.9],[0.8,0.2],[0.2,0.8],[0.7,0.3],[0.3,0.7],[0.6,0.4],[0.4,0.6],[0.65,0.35],[0.35,0.65],[0.75,0.25],[0.25,0.75],[0.85,0.15],[0.15,0.85],[0.95,0.05],[0.05,0.95],[0.99,0.01],[0.01,0.99],[0.999,0.001],[0.001,0.999]):
                        try:
                            q_GCMC = pyiast.iast(P_IAST * np.array([0.5,0.5]), [globals()[isotherm_1_GCMC_list[a]], globals()[isotherm_2_GCMC_list[b]]],adsorbed_mole_fraction_guess=fraction_guess)
                            S_G = q_GCMC[0]/q_GCMC[1]
                            break
                        except:
                            continue
                    else:
                        q_GCMC=['error','error']
                        S_G='error'

                    #append it to the list
                    L_GCMC_1.append(q_GCMC[0])
                    L_GCMC_2.append(q_GCMC[1])
                    S_GCMC.append(S_G)



            for a in range(10):
                for b in range(10):
                    for fraction_guess in ([0.5,0.5],[0.9,0.1],[0.1,0.9],[0.8,0.2],[0.2,0.8],[0.7,0.3],[0.3,0.7],[0.6,0.4],[0.4,0.6],[0.65,0.35],[0.35,0.65],[0.75,0.25],[0.25,0.75],[0.85,0.15],[0.15,0.85],[0.95,0.05],[0.05,0.95],[0.99,0.01],[0.01,0.99],[0.999,0.001],[0.001,0.999]):
                        try:
                            q_ML = pyiast.iast(P_IAST * np.array([0.5,0.5]), [globals()[isotherm_1_ML_list[a]], globals()[isotherm_2_ML_list[b]]],adsorbed_mole_fraction_guess=fraction_guess)
                            S_M = q_ML[0]/q_ML[1]
                            break
                        except:
                            continue
                    else:
                        q_ML=['error','error']
                        S_M='error'

                    #append it to the list
                    L_ML_1.append(q_ML[0])
                    L_ML_2.append(q_ML[1])
                    S_ML.append(S_M)

            pyiast_results.iloc[i,6]=L_GCMC_1
            pyiast_results.iloc[i,7]=L_GCMC_2
            pyiast_results.iloc[i,8]=S_GCMC

            pyiast_results.iloc[i,9]=L_ML_1
            pyiast_results.iloc[i,10]=L_ML_2
            pyiast_results.iloc[i,11]=S_ML

    return pyiast_results