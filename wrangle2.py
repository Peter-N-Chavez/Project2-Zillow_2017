import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import urllib.request
from PIL import Image
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures

import env
from explore2 import *

pd.options.mode.chained_assignment = None

def dtypes_to_list(df):

    num_type_list, cat_type_list = [], []

    for column in df:

        col_type =  df[column].dtype

        if col_type == "object" : 
        
            cat_type_list.append(column)
    
        if np.issubdtype(df[column], np.number) and \
             ((df[column].max() + 1) / df[column].nunique())  == 1 :

            cat_type_list.append(column)

        if np.issubdtype(df[column], np.number) and \
            ((df[column].max() + 1) / df[column].nunique()) != 1 :

            num_type_list.append(column)

    return num_type_list, cat_type_list

def create_dummies(df, object_cols):
    """
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns.
    It then appends the dummy variables to the original dataframe.
    It returns the original df with the appended dummy variables.
    """

    # run pd.get_dummies() to create dummy vars for the object columns.
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)

    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df

def get_db_url(hostname, username, password, database):
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url

def acquire_zillow():
    url_zillow = get_db_url(env.hostname, env.username, env.password, "zillow")
    query = pd.read_sql("""SELECT parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet AS cal_fin_sqf, taxvaluedollarcnt AS tax_val, yearbuilt AS year_built, taxamount, fips \
                        FROM properties_2017 \
                        JOIN predictions_2017 \
                            USING (parcelid) \
                        WHERE propertylandusetypeid = 261 \
                        OR propertylandusetypeid = 279 \
                        AND transactiondate LIKE '%%%%2017%%' \
                        AND taxdelinquencyflag NOT LIKE '%%%%%%Y%%' \
                        AND unitcnt NOT IN (2,3)
                        ;  \
                        """, url_zillow)

    filename = "zillow_predictions.csv"
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = query
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    df = df[df.tax_val < 900000]
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # removing outliers
    df = remove_outliers(df, 1.5, ['bedroomcnt', 'bathroomcnt', 'cal_fin_sqf', 'tax_val',           'taxamount'])
    
    # converting column datatypes
    df.fips = df.fips.round(0).astype(int)
    df.year_built = df.year_built.astype(object)
    df.tax_val = df.tax_val.astype(float)
    df.tax_val = df.tax_val.round(2)
    df.cal_fin_sqf = df.cal_fin_sqf.astype(float)

    df['fips'] = df.fips.map({6037 : "f6037", 6059 : "f6059", 6111 : "f6111"})
    dummies = pd.get_dummies(df['fips'],drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=249)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=249)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])

    train = train.dropna()
    validate = validate.dropna()
    test = test.dropna()

    train.year_built = train.year_built.astype(str)
    train.year_built = train.year_built.str.strip().str[0:4]
    train.year_built = train.year_built.astype(int)
    train.year_built = train.year_built.round(0)  
    
    validate.year_built = validate.year_built.astype(str)
    validate.year_built = validate.year_built.str.strip().str[0:4]
    validate.year_built = validate.year_built.astype(int)
    validate.year_built = validate.year_built.round(0)     

    test.year_built = test.year_built.astype(str)
    test.year_built = test.year_built.str.strip().str[0:4]
    test.year_built = test.year_built.astype(int)
    test.year_built = test.year_built.round(0)  
    
    return train, validate, test    

def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test
