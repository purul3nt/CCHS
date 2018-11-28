# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:06:33 2018

@author: Hannah
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:57:27 2018

@author: Hannah
"""


import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy 
from featexp import get_univariate_plots
# Importing the dataset
DATA_PATH = 'C:/Users/Hannah/Documents/Thesis/'
# Importing the dataset
#import csv
#with open('cchs_2014.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#dataset = read.csv('cchs_2014.csv')

def load_cchs_data(data_path=DATA_PATH):
    csv_path = os.path.join(data_path, 'cchs_2014.csv')
    return pd.read_csv(csv_path, low_memory=False)

cchsdata = load_cchs_data()
#cchsdata.info()

#data imputation

# mark NA values as NaN
cchsdata_1 = cchsdata[['GEN_02A2','PACDTLE',
'HUIDHSI',
'SPSDCON',
'SPSDATT',
'SPSDWOR',
'DISDK6',
'DISDDSX',
'LOPG100',
'INCDRCA',
'INCDRRS',
'GEN_02A2',
]].replace([99,96,97,98,99.9,99.0,99.00,99.999,99.000000], numpy.NaN)


cchsdata_2 = cchsdata[['HWTGBMI',
'LBSGHPW',
'FVCDTOT',
'PAC_2A',
'SMK_204',
'SMK_05B',
'SMKDYCS',
'ALWDDLY'
]].replace([999.99,996,997,998,999.9,999.0], numpy.NaN)



cchsdata_3 = cchsdata[['GEN_02B',
'GEN_08',
'GEN_10',
'GENDHDI',
'GENDMHI',
'GENGSWL',
'SLP_02',
'CCC_031',
'CCC_051',
'CCC_071',
'CCC_091',
'CCC_101',
'CCC_121',
'CCC_131',
'CCC_31A',
'CCC_151',
'CCC_280',
'CCC_290',
'HCU_1AA',
'RAC_1',
'CCS_180',
'PAC_3A',
'PAC_3J',
'PACDPAI',
'SMK_202',
'SPS_01',
'SPS_08',
'LBSDWSS',
'LBSGSOC',
'EDUDR04',
'SDCGCB13',
'SDCFIMM',
'INCGHH','GEN_01']].replace([6,7,8,9], numpy.NaN)

cchsdata_4 = cchsdata[['ADM_RNO',
'GEOGPRV',
'GEODPMF',
'DHHGAGE',
'DHH_SEX',
]]

#join all columns
cchsdata_with_NaN = pd.concat([cchsdata_1, cchsdata_2,cchsdata_3,cchsdata_4], axis=1, join='inner')


#outputting descriptive information about the data in a csv report
df = cchsdata_with_NaN.describe(include = 'all')
df.to_csv('describe_columns_with_NaN.csv')
#show description
df


#fill NaN with mean

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='most_frequent',axis=0) 
imputer.fit(cchsdata_with_NaN) 
imputer.statistics_
##imputing data with central tendancy mean 
cchsdata_imputed = imputer.transform(cchsdata_with_NaN)
#re attaching headers
cchsdata_imputed = pd.DataFrame(cchsdata_imputed, columns = cchsdata_with_NaN.columns)
##did not work imputed_cchs_mean = cchsdata_with_NaN.fillna(cchsdata_with_NaN.mean())

countcchs = cchsdata.apply(pd.value_counts)
countcchs
countcchs.plot(countcchs)

plt.savefig('countcchs.pdf')




cchsdata.hist(bins=50, figsize=(20,15))
plt.show()

# Plots drawn for all features if nothing is passed in feature_list parameter.
bins=[0.5,0.75,1.0,1.5,2.0,2.5,3.0]
uni_plots = get_univariate_plots(data=cchsdata, target_col='CCC_290', bins = 3)
#scatter plot WIP
from pandas.plotting import scatter_matrix
scatter_matrix(cchsdata_imputed['CCC_290'], figsize=(12, 8))

#Trends between features for test and trainthis needs to be compared with test data

from featexp import get_trend_stats
stats = get_trend_stats(data=cchsdata, target_col='CCC_290', bins = 3)


stats
# Splitting the dataset into the Training set and Test set (note: might want to do stratified sampling)
# splitting variables where Y is the target variable and X is the inpt
X = cchsdata_imputed.loc[:, df.columns != 'CCC_290']
y = cchsdata_imputed['CCC_290']





# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

##resampling training sets (only!!!)
alltrainingset = pd.concat([X_train,y_train], axis=1)

#Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##changing target into categorical int value for Y TRAIN
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
encoded_y_train = lab_enc.fit_transform(y_train)



##changing target into categorical int value for Y TEST
from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
encoded_y_test = lab_enc.fit_transform(y_test)

#Remove features with low variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.99 * (1 - .99)))
sel.fit_transform(X_train)


#Remove features with low variance for X_TEST
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.99 * (1 - .99)))
sel.fit_transform(X_test)



#Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train, encoded_y_train)
clf.feature_importances_ 
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X_train)



#Tree-based feature selection for X_TEST
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_test, encoded_y_test)
clf.feature_importances_ 
model = SelectFromModel(clf, prefit=True)
X_test_new = model.transform(X_test)




#K-NN
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=7, weights='distance')  
classifier.fit(X_new, encoded_y_train)  

# Predicting the Test set results
y_pred = classifier.predict(X_test_new)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(encoded_y_test, y_pred)

#K NN Evaluation
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(encoded_y_test, y_pred))  
print(classification_report(encoded_y_test, y_pred))

#calculating the error to adjust the K value
error = []

# Calculating error for K values between 1 and 40 (since n_neighbours = 5 was just a guess)
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_new, encoded_y_train)
    pred_i = knn.predict(X_test_new)
    error.append(numpy.mean(pred_i != encoded_y_test)) 
    
    plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  
plt.savefig('KNNPLOT.pdf')
##################NEED TO DOWNSAMPLE

