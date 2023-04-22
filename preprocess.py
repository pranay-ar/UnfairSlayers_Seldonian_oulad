import os
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from seldonian.utils.io_utils import save_json

import matplotlib.pyplot as plt

file = 'studentInfo.csv'

cols_org = ["code_module","code_presentation","id_student","gender","region","highest_education","imd_band","age_band",
            "num_of_prev_attempts","studied_credits","disability","final_result"]

def make_seldonian_dataset(input_path,output_path_data,output_path_metadata):
    """ load the dataset into features and label arrays.
    One-hot encode categorical features,
    scale numerical features to have unit variance and 0 mean
    and then encode label column to have binary output: 
    0: good credit and 1: bad credit. 
    Finally, save resulting dataframe to a CSV file
    and make metadata JSON file
    
    :param input_path: The path to the original dataset in CSV format
    :type input_path: str
    
    :param output_path_data: The filename for saving the reformated dataset file 
    :type output_path_data: str

    :param output_path_metadata: The filename for saving the reformated metadata file 
    :type output_path_metadata: str
    """
    
    df = pd.read_csv(input_path,header=None,names=cols_org)
    
    # split into inputs and outputs
    X = df.drop(columns=["final_result"])
    y = df["final_result"]
    
    # drop columns like student id
    X.drop(columns=["id_student"],inplace=True)

    # convert distinction and pass to 1 and fail and withdraw to 0
    y = y.str.replace('Distinction','1')
    y = y.str.replace('Pass','1')
    y = y.str.replace('Fail','0')
    y = y.str.replace('Withdrawn','0')

    # convert studied_credits to int    
    X['studied_credits'] = pd.to_numeric(X['studied_credits'],errors='coerce')
    X['num_of_prev_attempts'] = pd.to_numeric(X['num_of_prev_attempts'],errors='coerce')

    # select categorical features and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns

    # one hot encode cat features only, scale numerical features using standard scaler 
    ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',StandardScaler(),num_ix)])
    
    # Apply transformation
    X = ct.fit_transform(X)
    print("The shape of X is",X.shape)
    # label encode the target variable to have the classes 0 and 1
    # 0 is good credit, 1 is bad credit
    y = LabelEncoder().fit_transform(y)
    
    # Get names after one-hot encoding
    output_columns = ct.get_feature_names_out(ct.feature_names_in_)
    print(f"Output columns: {output_columns.shape}")
    # Make an output dataframe to save from X and y
    print("the types of X and y are",type(X),type(y))
    # convert x into an ndarray
    X = X.toarray()
    print("the types of X and y are",type(X),type(y))
    outdf = pd.DataFrame(X,columns=output_columns)
    
    # Change name of the two one-hot encoded sex columns to M and F
    outdf.rename(columns={'c__gender_F':'F','c__gender_M':'M'},inplace=True)
    outdf.rename(columns={'c__disability_N':'N','c__disability_Y':'Y'},inplace=True)
    # outdf.rename(columns={'c__gender_F':'F','c__gender_M':'M'},inplace=True)
    # Add label column into final dataframe
    outdf['final_result'] = y
    
    # Save final dataframe
    outdf.to_csv(output_path_data,index=False,header=False)
    print(f"Saved data file to: {output_path_data}")
    print()
    
    # Save metadata json file
    metadata_dict = {
        "regime":"supervised_learning",
        "sub_regime":"classification",
        "all_col_names":list(outdf.columns),
        "label_col_names":"final_result",
        "sensitive_col_names":["F","M","N","Y"]
    }
    
    with open(output_path_metadata,'w') as outfile:
        json.dump(metadata_dict,outfile,indent=2)
    print(f"Saved metadata file to: {output_path_metadata}")
    return 

make_seldonian_dataset(
    input_path= file,
    output_path_data="studentInfoconverted.csv",
    output_path_metadata="metadata_studentInfo.json")