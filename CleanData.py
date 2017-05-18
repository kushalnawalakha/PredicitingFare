from math import sin, cos, sqrt, atan2, radians
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil import parser
from sklearn import datasets, linear_model

# approximate radius of earth in km
from sklearn.ensemble import RandomForestRegressor
listOfColumnsInConsideration=["tolls_amount", "tip_amount", "mta_tax", "rate_code",
             "payment_type", "surcharge","pickup_longitude","pickup_latitude","pickup_hour","total_time"]

testset=pd.read_csv("Data/test.csv",sep=",")
dataset=pd.read_csv("Data/train.csv",sep=",")



print(testset.describe())
#dataset=dataset.sample(frac=0.001)
print("read complete")
surcharge_mean=np.mean(dataset["surcharge"].loc[(dataset.surcharge.notnull())])

def copy(row):
    if np.isnan(row['pickup_latitude']):
        row['pickup_latitude']=row['dropoff_latitude']
    if np.isnan(row['dropoff_latitude']):
        row['dropoff_latitude']=row['pickup_latitude']
    if np.isnan(row['dropoff_longitude']):
        row['dropoff_longitude']=row['pickup_longitude']
    if np.isnan(row['pickup_longitude']):
        row['pickup_longitude']=row['dropoff_longitude']
    if np.isnan(row['tip_amount']):
        row['tip_amount']=0
    if np.isnan(row['surcharge']) or row['surcharge']<0:
        row['surcharge'] = surcharge_mean
    if row["pickup_datetime"]!='':
        row["pickup_hour"]=parser.parse(row["pickup_datetime"]).hour
    if row["pickup_datetime"]!="" and row["dropoff_datetime"]!="":
        row["total_time"]=(parser.parse(row["dropoff_datetime"]) - parser.parse(row["pickup_datetime"])).seconds/60
    return row


dataset['new_user']=dataset['new_user'].map({'YES': 1, 'NO': 0})
dataset['payment_type']=dataset['payment_type'].map({'CSH': 1, 'CRD': 0})
print(dataset.corr())
dataset=dataset.loc[(dataset['new_user'].notnull())]
dataset=dataset.apply(lambda row: copy(row),axis=1)
dataset=dataset.dropna(subset=["dropoff_longitude","pickup_latitude"])
print(dataset.describe().T)

t_train = dataset.loc[(dataset.payment_type.notnull())].sample(frac=0.1,replace=False)
t_test = dataset.loc[(dataset.payment_type.isnull())]
t_X = t_train[["tip_amount","surcharge","mta_tax"]]
t_Y = t_train["payment_type"]
t_rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1).fit(t_X, t_Y)
t_pred = t_rtr.predict(t_test[["tip_amount","surcharge","mta_tax"]])
dataset.loc[(dataset.payment_type.isnull()), 'payment_type'] =t_pred
dataset.to_csv("cleanup_train.csv",sep=",")

print(dataset.corr())



#clf = SGDClassifier(loss="hinge", penalty="l2")


regr = linear_model.SGDRegressor (alpha =0.00001, penalty="l2")
X = dataset[listOfColumnsInConsideration]
Y = dataset["fare_amount"]
regr.fit(X, Y)

testset=testset.apply(lambda row: copy(row),axis=1)
testset.loc[testset.pickup_latitude.isnull(),'pickup_latitude']=np.mean(testset["pickup_latitude"])
testset.loc[testset.pickup_longitude.isnull(),'pickup_longitude']=np.mean(testset["pickup_longitude"])
testset['payment_type']=testset['payment_type'].map({'CSH': 1, 'CRD': 0})
testset['new_user']=testset['new_user'].map({'YES': 1, 'NO': 0})
testset.loc[(testset['new_user'].isnull()), 'new_user'] =0
t_testset=testset[testset.payment_type.isnull()]
t_pred = t_rtr.predict(t_testset[listOfColumnsInConsideration])
testset.loc[(testset.payment_type.isnull()), 'payment_type'] =t_pred
dataset.to_csv("cleanup_test.csv",sep=",")
print("clean up complete")

print(testset.describe().T)

predictions=regr.predict(testset[listOfColumnsInConsideration])
testset["fare_amount"]=predictions
testset[["TID","fare_amount"]].to_csv("outcome.csv",sep=",")

d=dataset.sample(frac=0.1, replace=False)
X=d[listOfColumnsInConsideration]
Y=dataset["fare_amount"]

predictions=regr.predict(X)
d["predictions"]=predictions
d["actual"]=Y
d.to_csv("outcome_random.csv",sep=",")