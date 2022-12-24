# IMPORT AND FUNCTIONS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.model_selection import KFold   
from statistics import mean
import joblib 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# GET THE DATA (DONE). LOAD DATA

data = pd.read_csv('data/Nha.csv')

# DISCOVER THE DATA TO GAIN INSIGHTS
# 3.1 Quick view of the data

'''print('\n____________ Dataset info ____________')
print(data.info())              
print('\n____________ Some first data examples ____________')
print(data.head(3)) 
print('\n____________ Counts on a feature ____________')
print(data['Diện tích - m2'].value_counts()) 
print('\n____________ Statistics of numeric features ____________')
print(data.describe())    
print('\n____________ Get specific rows and cols ____________')     
print(data.iloc[[0,5,48], [2, 5]] ) 
# Refer using column ID

# 3.2 Scatter plot b/w 2 features
if 1:
    data.plot(kind="scatter", y="Mức giá - Tỷ", x="Số phòng ngủ", alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    plt.savefig('figures/Nha/scatter_1_feat.png', format='png', dpi=300)
    plt.show()      
if 1:
    data.plot(kind="scatter", y="Mức giá - Tỷ", x="Diện tích - m2", alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    plt.savefig('figures/Nha/scatter_2_feat.png', format='png', dpi=300)
    plt.show()

# 3.3 Scatter plot b/w every pair of features
if 1:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["Mức giá - Tỷ", "Số phòng ngủ", "Số toilet", "Diện tích - m2"]
    scatter_matrix(data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig('figures/Nha/scatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

# 3.4 Plot histogram of 1 feature
if 1:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["Mức giá - Tỷ"]
    scatter_matrix(data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig('figures/Nha/histogram_1_feature.png', format='png', dpi=300)
    plt.show()

# 3.5 Plot histogram of numeric features
if 1:
    #data.hist(bins=10, figsize=(10,5)) #bins: no. of intervals
    data.hist(figsize=(10,5)) #bins: no. of intervals
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.savefig('figures/Nha/hist_data.png', format='png', dpi=300) # must save before show()
    plt.show()


# 3.6 Compute correlations b/w features
corr_matrix = data.corr()
print(corr_matrix) # print correlation matrix
print('\n',corr_matrix["Mức giá - Tỷ"].sort_values(ascending=False)) 
# print correlation b/w a feature and other features

# 3.7 Try combining features
data["DIỆN TÍCH PHÒNG"] = data["Diện tích - m2"] / data["Số phòng ngủ"] 
data["TỔNG SỐ PHÒNG"] = data["Số phòng ngủ"] + data["Số toilet"] 
corr_matrix = data.corr()
print(corr_matrix["Mức giá - Tỷ"].sort_values(ascending=False)) 
# print correlation b/w a feature and other features
data.drop(columns = ["DIỆN TÍCH PHÒNG", "TỔNG SỐ PHÒNG"], inplace=True) 
# remove experiment columns'''


# PREPARE THE DATA 

# 4.1 Remove unused features
data.drop(columns = ["STT", "Nhu cầu"], inplace=True) 

# 4.2 Split training-test set and NEVER touch test set until test phase
method = 2

if method == 1: 
    # Method 1: Randomly select 20% of data for test set. Used when data set is large
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42) 
    # set random_state to get the same training set all the time, 
    # otherwise, when repeating training many times, your model may see all the data
elif method == 2: 
    # Method 2: Stratified sampling, to remain distributions of important features, see (Geron, 2019) page 56
    # Create new feature "KHOẢNG GIÁ": the distribution we want to remain
    data["KHOẢNG GIÁ"] = pd.cut(data["Mức giá - Tỷ"],
                                    bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, np.inf],
                                    #labels=["<10 tỷ", "10-20 tỷ", "20-30 tỷ", "30-40 tỷ", "40-50 tỷ", "50-60 tỷ","60-70 tỷ","70-80 tỷ","80-90 tỷ","90-100 tỷ","100-110 tỷ" , "110-120 tỷ"">120 tỷ"])
                                    labels=[10,20,30,40,50,60,70,80,90,100,110,120,600]) 
                                    # use numeric labels to plot histogram
    
    # Create training and test set
    from sklearn.model_selection import StratifiedShuffleSplit  
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
    # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
    # (if you want to run the algorithm n_splits times with different train-test set)
    
    for train_index, test_index in splitter.split(data, data["KHOẢNG GIÁ"]): 
        # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = data.iloc[train_index]
        test_set = data.iloc[test_index]              
    
    # See if it worked as expected
    if 1:
        data["KHOẢNG GIÁ"].hist(bins=14, figsize=(5,5)); 
        plt.savefig('figures/Nha/data_hist_price.png', format='png', dpi=300)
        #plt.show();
        train_set["KHOẢNG GIÁ"].hist(bins=14, figsize=(5,5)); 
        plt.savefig('figures/Nha/train_set_hist_price.png', format='png', dpi=300)
        #plt.show()

    # Remove the new feature
    #print(train_set.info())
    for _set_ in (train_set, test_set):
        #_set_.drop("income_cat", axis=1, inplace=True) # axis=1: drop cols, axis=0: drop rows
        _set_.drop(columns="KHOẢNG GIÁ", inplace=True) 
    
'''
print('\n____________ Split training and test set ____________')     
print(len(train_set), "training +", len(test_set), "test examples")
print(train_set.head(4))'''


# 4.3 Separate labels from data, since we do not process label values

train_set_labels = train_set["Mức giá - Tỷ"].copy()
train_set = train_set.drop("Mức giá - Tỷ" , axis=1) 
test_set_labels = test_set["Mức giá - Tỷ"].copy()
test_set = test_set.drop("Mức giá - Tỷ", axis=1) 

# 4.4 Define pipelines for processing data. 
# 4.4.1 Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         

num_feat_names = ['Năm' , 'Diện tích - m2', 'Số lầu' , 'Mặt tiền - m', 'Đường vào - m', 'Số phòng ngủ' , 'Số toilet'] 
# =list(train_set.select_dtypes(include=[np.number]))
cat_feat_names = ['Quận' , 'Phường' , 'Đường' , 'Hướng nhà' , 'Hướng ban công', 'Nội thất' , 'Pháp lý'] 
# =list(train_set.select_dtypes(exclude=[np.number])) 

# 4.4.2 Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
    ])    

# 4.4.3 Define MyFeatureAdder: a transformer for adding features "TỔNG SỐ PHÒNG",...  
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_TONG_SO_PHONG = True): 
        # MUST NO *args or **kargs
        self.add_TONG_SO_PHONG = add_TONG_SO_PHONG
    def fit(self, feature_values, labels = None):
        return self  # nothing to do here
    def transform(self, feature_values, labels = None):
        if self.add_TONG_SO_PHONG:        
            SO_PHONG_id, SO_TOILETS_id = 5, 6
            # column indices in num_feat_names. can't use column names b/c the transformer SimpleImputer removed them
            # NOTE: a transformer in a pipeline ALWAYS return dataframe.values (ie., NO header and row index)
            
            TONG_SO_PHONG = feature_values[:, SO_PHONG_id] + feature_values[:, SO_TOILETS_id]
            feature_values = np.c_[feature_values, TONG_SO_PHONG] 
            #concatenate np arrays
        return feature_values

# 4.4.4 Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), 
    # copy=False: imputation will be done in-place 
    ('attribs_adder', MyFeatureAdder(add_TONG_SO_PHONG = True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) 
    # Scale features to zero mean and unit variance
    ])  

# 4.4.5 Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])

# 4.5 Run the pipeline to process training data           
processed_train_set_val = full_pipeline.fit_transform(train_set)
'''print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(num_feat_names)))
'''
joblib.dump(full_pipeline, r'models/full_pipeline_2.pkl')


param_grid = [
            # try 12 (3x4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 39]},
            # then try 12 (4x3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]} ]

# TRAIN AND EVALUATE MODELS
'''
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(processed_train_set_val, train_set_labels)

final_model = grid_search.best_estimator_'''

def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_GridSearch_CV' '_model_2.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('models/' + model_name + '_GridSearch_CV' + '_model_2.pkl')
    #print(model)
    return model

#store_model(final_model)
final_model = load_model("RandomForestRegressor")

# Prediction
some_data = train_set.iloc[:5]
some_labels = train_set_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
# Prediction 5 samples 
'''print("Predictions:", final_model.predict(some_data_prepared))
print("Labels:", list(some_labels))
print('\n')'''

def predict_input_user(data):
    
    row_label = [1]
    sample = pd.DataFrame(data=data,index=row_label)
    sample_prepared = full_pipeline.transform(sample)

    return final_model.predict(sample_prepared)
