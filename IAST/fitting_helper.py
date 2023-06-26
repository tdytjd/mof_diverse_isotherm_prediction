#pyIAST modified fitting with positive constraints
import pyIAST_single_isotherm_fitting_modified
import pyiast
import numpy as np
import pandas as pd
import json


def fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Langmuir",param_guess=None,loading_key="loading (mol/kg)",pressure_key="pressure (Pa)"):
    #fitting works
    if model=="Langmuir":
        bnds=((1e-5,np.inf),(1e-15,np.inf))
    elif model=="BET" or model=="Quadratic":
        bnds=((1e-5,np.inf),(1e-15,np.inf),(1e-15,np.inf))
    elif model=='DSLangmuir':
        bnds=((1e-5,np.inf),(1e-15,np.inf),(1e-5,np.inf),(1e-15,np.inf))
    elif model=='TemkinApprox':
        bnds=((1e-5,np.inf),(1e-15,np.inf),(-1,1))    
    
    try:
        #with bounds
        isotherm_1=pyIAST_single_isotherm_fitting_modified.ModelIsotherm(df_mol,loading_key=loading_key,
                            pressure_key=pressure_key,param_guess=param_guess,
                            model=model,param_bnds=bnds)
        #unreasonable fitting
        if (isotherm_1.loading(pressure_grid*max_pressure)<0).sum()>0:
            rmse_1=100001
            param_1=isotherm_1.params
        #reasonable fitting
        else:
            rmse_1=isotherm_1.rmse
            param_1=isotherm_1.params
 
    #fitting not working at all
    except:
        isotherm_1='error'
        rmse_1=1000000
        param_1=[]
        
    try:
        #without bounds
        isotherm_2=pyIAST_single_isotherm_fitting_modified.ModelIsotherm(df_mol,loading_key=loading_key,
                            pressure_key=pressure_key,param_guess=param_guess,
                            model=model)
        #unreasonable fitting
        if (isotherm_2.loading(pressure_grid*max_pressure)<0).sum()>0:
            rmse_2=100001
            param_2=isotherm_2.params
        #temkin approx
        elif 'theta' in isotherm_2.params:
            if abs(isotherm_2.params['theta'])>1 or isotherm_2.params['K']<0 or isotherm_2.params['M']<0:
                rmse_2=100001
                param_2=isotherm_2.params
            else:
                rmse_2=isotherm_2.rmse
                param_2=isotherm_2.params
        elif len([i for i in isotherm_2.params if isotherm_2.params[i]<0])>0:
            rmse_2=100001
            param_2=isotherm_2.params
        #reasonable fitting
        else:
            rmse_2=isotherm_2.rmse
            param_2=isotherm_2.params
 
    #fitting not working at all
    except:
        isotherm_2='error'
        rmse_2=1000000
        param_2=[]
    
    if rmse_2<rmse_1:
        isotherm=isotherm_2
        rmse=rmse_2
        param=param_2
    else:
        isotherm=isotherm_1
        rmse=rmse_1
        param=param_1
        
    return isotherm,rmse,param


def fit_single_isotherms_iast(df_mol,max_pressure,K_guess):
    #set a pressure vector to test whether the fitted isotherms are reasonable by looking at the loadings
    #df_mol is the ready dataframe for a given mof and mol
    #this is for fitting process in IAST calculations
    args = (np.arange(1e-8,1e-6,1e-8),np.arange(1e-6,1e-5,1e-7), np.arange(1e-5,1e-4,1e-6),np.arange(1e-4,1e-3,1e-5),np.arange(1e-3,1e-2,1e-4),np.arange(1e-2,1e-1,1e-3),np.arange(1e-1,1e0,1e-2),np.arange(1e0,1e1,1e-1),np.arange(1e1,1e2,1e0))
    pressure_grid = np.concatenate(args)
    
    #create a list to record the rmse
    rmse_list=[]
    
    #create a list to record params
    params_list=[]
    
    #langmuir
    Langmuir_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Langmuir")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    Langmuir_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Langmuir",param_guess={'K':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    
    #Quadratic
    Quadratic_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Quadratic")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    Quadratic_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Quadratic",param_guess={'Ka':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    

    #BET
    BET_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="BET")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    BET_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="BET",param_guess={'Ka':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    

    #DS langmuir
    DSLangmuir_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="DSLangmuir")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    DSLangmuir_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="DSLangmuir",param_guess={'K1':K_guess*0.5,'K2':K_guess*0.01})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    

    ##temkin approx
    TemkinApprox_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="TemkinApprox")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    TemkinApprox_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="TemkinApprox",param_guess={'K':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    
        
    def compare_langmuir_with_other(isotherm_to_be_compared,min_index,min_value,isotherm_model_name):
        model_params=[isotherm_to_be_compared.params]
        smaller_langmuir=min(rmse_list[0],rmse_list[1])
        min_lang_index = rmse_list.index(smaller_langmuir)
        langmuir_compare_signal=0
        isotherm_model=isotherm_to_be_compared

        if smaller_langmuir<100:
            langmuir_compare_signal=1
        
        if langmuir_compare_signal==1:
            if min_lang_index==0:
                langmuir_model=Langmuir_isotherm_1
                langmuir_params=Langmuir_isotherm_1.params
                langmuir_loading=np.array(Langmuir_isotherm_1.loading(np.array(df_mol['pressure (Pa)'])))
            elif min_lang_index==1:
                langmuir_model=Langmuir_isotherm_2
                langmuir_params=Langmuir_isotherm_2.params
                langmuir_loading=np.array(Langmuir_isotherm_2.loading(np.array(df_mol['pressure (Pa)'])))

            test_loading=np.array(isotherm_to_be_compared.loading(np.array(df_mol['pressure (Pa)'])))
            if (((abs(test_loading-langmuir_loading)/test_loading)<0.025).sum()==test_loading.size):
                isotherm_model_name='Langmuir*'
                model_params=[langmuir_params]
                min_index=min_lang_index
                min_value=smaller_langmuir
                isotherm_model=langmuir_model
                
        return isotherm_model,min_index,min_value,isotherm_model_name,model_params
    
    
    min_value = min(rmse_list)
    min_index = rmse_list.index(min_value)
    


    #if none of the model fits or the fitting is bad (rmse>1), do linear interpolation
    #for simplicity, use highest loading as the fill value
    if min_value>1:
        isotherm_model_name='Linear'
        min_value=None
        min_index=None
        model_params=None
        isotherm_model=pyIAST_single_isotherm_fitting_modified.InterpolatorIsotherm(df_mol,
                                    loading_key="loading (mol/kg)",
                                    pressure_key="pressure (Pa)",fill_value=df_mol.iloc[0,1])
    elif min_index==0:
        isotherm_model_name='Langmuir'
        model_params=[Langmuir_isotherm_1.params]
        isotherm_model=Langmuir_isotherm_1
    elif min_index==1:
        isotherm_model_name='Langmuir'
        model_params=[Langmuir_isotherm_2.params]
        isotherm_model=Langmuir_isotherm_2
    elif min_index==2:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(Quadratic_isotherm_1,min_index,min_value,"Quadratic")
    elif min_index==3:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(Quadratic_isotherm_2,min_index,min_value,"Quadratic")
    elif min_index==4:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(BET_isotherm_1,min_index,min_value,"BET")
    elif min_index==5:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(BET_isotherm_2,min_index,min_value,"BET")
    elif min_index==6:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(DSLangmuir_isotherm_1,min_index,min_value,"DSLangmuir")
    elif min_index==7:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(DSLangmuir_isotherm_2,min_index,min_value,"DSLangmuir")
    elif min_index==8:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(TemkinApprox_isotherm_1,min_index,min_value,"TemkinApprox")
    elif min_index==9:
        isotherm_model,min_index,min_value,isotherm_model_name,model_params=compare_langmuir_with_other(TemkinApprox_isotherm_2,min_index,min_value,"TemkinApprox")
    
    return isotherm_model#rmse_list,min_index,min_value,isotherm_model_name,model_params,params_list
        
    

    
    
    

    
def fit_single_isotherms(MOFs_name,mol_name,data):
    #set a pressure vector to test whether the fitted isotherms are reasonable by looking at the loadings
    #data is the whole json file
    args = (np.arange(1e-8,1e-6,1e-8),np.arange(1e-6,1e-5,1e-7), np.arange(1e-5,1e-4,1e-6),np.arange(1e-4,1e-3,1e-5),np.arange(1e-3,1e-2,1e-4),np.arange(1e-2,1e-1,1e-3),np.arange(1e-1,1e0,1e-2),np.arange(1e0,1e1,1e-1),np.arange(1e1,1e2,1e0))
    pressure_grid = np.concatenate(args)
    
    #create a dataframe for pyIAST
    number_of_points=len(data[MOFs_name][mol_name]['pressure (Pa)'])
    df_mol=pd.DataFrame(columns=['pressure (Pa)','loading (mol/kg)'],index=range(number_of_points),dtype='object')
    for k in range(number_of_points):
        df_mol.iloc[k,0]=data[MOFs_name][mol_name]['pressure (Pa)'][k]
        df_mol.iloc[k,1]=data[MOFs_name][mol_name]['loading (mol/kg)'][k]
    df_mol=df_mol.astype(float)
    max_pressure=max(data[MOFs_name][mol_name]['pressure (Pa)'])
    max_loading=max(data[MOFs_name][mol_name]['loading (mol/kg)'])
        
    #get the KH of this pair, and divide it by max loading
    if max_loading!=0:
        K_guess=data[MOFs_name][mol_name]['KH (mol/(kgPa))']/max_loading
    else:
        K_guess=data[MOFs_name][mol_name]['KH (mol/(kgPa))']

    #create a list to record the rmse
    rmse_list=[]
    
    #create a list to record params
    params_list=[]
    
    #langmuir
    Langmuir_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Langmuir")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    Langmuir_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Langmuir",param_guess={'K':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    
    #Quadratic
    Quadratic_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Quadratic")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    Quadratic_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="Quadratic",param_guess={'Ka':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    

    #BET
    BET_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="BET")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    BET_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="BET",param_guess={'Ka':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    

    #DS langmuir
    DSLangmuir_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="DSLangmuir")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    DSLangmuir_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="DSLangmuir",param_guess={'K1':K_guess*0.5,'K2':K_guess*0.01})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    

    ##temkin approx
    TemkinApprox_isotherm_1,rmse_1,params_1=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="TemkinApprox")
    rmse_list.append(rmse_1)  
    params_list.append(params_1)
    
    TemkinApprox_isotherm_2,rmse_2,params_2=fit_isotherm_given_model(df_mol,pressure_grid,max_pressure,model="TemkinApprox",param_guess={'K':K_guess})
    rmse_list.append(rmse_2)  
    params_list.append(params_2)
    
        
    def compare_langmuir_with_other(isotherm_to_be_compared,min_index,min_value,isotherm_model):
        model_params=[isotherm_to_be_compared.params]
        smaller_langmuir=min(rmse_list[0],rmse_list[1])
        min_lang_index = rmse_list.index(smaller_langmuir)
        langmuir_compare_signal=0
        if smaller_langmuir<100:
            langmuir_compare_signal=1
        
        if langmuir_compare_signal==1:
            if min_lang_index==0:
                langmuir_params=Langmuir_isotherm_1.params
                langmuir_loading=np.array(Langmuir_isotherm_1.loading(np.array(data[MOFs_name][mol_name]['pressure (Pa)'])))
            elif min_lang_index==1:
                langmuir_params=Langmuir_isotherm_2.params
                langmuir_loading=np.array(Langmuir_isotherm_2.loading(np.array(data[MOFs_name][mol_name]['pressure (Pa)'])))

            test_loading=np.array(isotherm_to_be_compared.loading(np.array(data[MOFs_name][mol_name]['pressure (Pa)'])))
            if (((abs(test_loading-langmuir_loading)/test_loading)<0.025).sum()==test_loading.size):
                isotherm_model='Langmuir*'
                model_params=[langmuir_params]
                min_index=min_lang_index
                min_value=smaller_langmuir
                
        return min_index,min_value,isotherm_model,model_params
    
    
    min_value = min(rmse_list)
    min_index = rmse_list.index(min_value)
    


    #if none of the model fits, do linear interpolation
    if min_value>1:
        isotherm_model='Linear'
        min_value=None
        min_index=None
        model_params=None
    elif min_index==0:
        isotherm_model='Langmuir'
        model_params=[Langmuir_isotherm_1.params]
    elif min_index==1:
        isotherm_model='Langmuir'
        model_params=[Langmuir_isotherm_2.params]
    elif min_index==2:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(Quadratic_isotherm_1,min_index,min_value,"Quadratic")
    elif min_index==3:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(Quadratic_isotherm_2,min_index,min_value,"Quadratic")
    elif min_index==4:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(BET_isotherm_1,min_index,min_value,"BET")
    elif min_index==5:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(BET_isotherm_2,min_index,min_value,"BET")
    elif min_index==6:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(DSLangmuir_isotherm_1,min_index,min_value,"DSLangmuir")
    elif min_index==7:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(DSLangmuir_isotherm_2,min_index,min_value,"DSLangmuir")
    elif min_index==8:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(TemkinApprox_isotherm_1,min_index,min_value,"TemkinApprox")
    elif min_index==9:
        min_index,min_value,isotherm_model,model_params=compare_langmuir_with_other(TemkinApprox_isotherm_2,min_index,min_value,"TemkinApprox")
    
    return rmse_list,min_index,min_value,isotherm_model,model_params,params_list
        
    


    