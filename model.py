import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_train=pd.read_excel('Data_Train.xlsx')

df_train.head()

df_train.info()

df_train.describe()

df_train.columns

df_train['Total_Stops'].value_counts()

df_train.shape

df_train['Duration'].value_counts()

df_train.isnull().sum()

df_train.dropna(inplace=True)

df_train.isnull().sum()

df_train.shape

df_train['Journey_Day']=pd.to_datetime(df_train.Date_of_Journey,format='%d/%m/%Y').dt.day

df_train['Journey_Month']=pd.to_datetime(df_train.Date_of_Journey,format='%d/%m/%Y').dt.month

df_train['Journey_Year']=pd.to_datetime(df_train.Date_of_Journey,format='%d/%m/%Y').dt.year

df_train.info()

df_train.describe()

df_train['Journey_Year'].value_counts()

df_train.drop('Date_of_Journey',axis=1,inplace=True)

df_train.info()

df_train.shape

df_train["Dep_hour"] = pd.to_datetime(df_train["Dep_Time"]).dt.hour

df_train["Dep_min"] = pd.to_datetime(df_train["Dep_Time"]).dt.minute

df_train.drop('Dep_Time',axis=1,inplace=True)

df_train.info()

df_train.head()

df_train["Arrival_hour"] = pd.to_datetime(df_train.Arrival_Time).dt.hour

df_train["Arrival_min"] = pd.to_datetime(df_train.Arrival_Time).dt.minute

df_train.drop(["Arrival_Time"], axis = 1, inplace = True)

df_train.head()

duration=list(df_train['Duration'])
for i in range(len(duration)):
  if('h' not in duration[i] or 'm' not in duration[i]):
    if('h' not in duration[i]):
      duration[i]='0h '+ duration[i]
    else:
      duration[i]=duration[i] + ' 0m'
duration_hour=[]
duration_min=[]

for i in range(len(duration)):
  time=list(duration[i].split(' '))
  duration_hour.append(int(time[0][:-1]))
  duration_min.append(int(time[1][:-1]))

df_train['Duration_hour']=duration_hour
df_train['Duration_min']=duration_min
df_train.drop('Duration',axis=1,inplace=True)

df_train.head()

df_train['Total_Stops'].value_counts()

df_train['Total_Stops'].replace(['1 stop','non-stop','2 stops','3 stops','4 stops'],[int(1),int(0),int(2),int(3),int(4)],inplace=True)

df_train.info()

"""Handling Categorial Data"""



df_train['Airline'].sort_values()

#airline vs price
plt.figure(figsize=(16,10))
sns.boxplot(x='Airline',y=df_train['Price'],data=df_train.sort_values("Price", ascending = False))
plt.xticks(rotation = 90)
#plt.show()

"""we abserved that all the airline prices are normal but jetairlines have a higher price compared to others"""

# as airlines is Nominal Categorial data we will perform onehot encoding

Airline=pd.get_dummies(df_train[['Airline']],drop_first=True)
Airline.head()

df_train.info()

df_train['Source'].value_counts()

plt.figure(figsize=(16,10))
sns.boxplot(x='Source',y=df_train['Price'].sort_values(),data=df_train.sort_values("Price", ascending = False))

#plt.show()

# as Source is Nominal Categorial data we will perform onehot encoding
Source=df_train[['Source']]
Source=pd.get_dummies(Source,drop_first=True)
Source.head()

df_train['Destination'].value_counts()

plt.figure(figsize=(16,10))
sns.boxplot(x='Destination',y=df_train['Price'].sort_values(),data=df_train.sort_values("Price", ascending = False))
plt.xticks(rotation = 90)
#plt.show()

# as destination is Nominal Categorial data we will perform onehot encoding

Destination=pd.get_dummies(df_train[['Destination']],drop_first=True)
Destination.head()

df_train['Route']

df_train.drop(['Route','Additional_Info'],axis=1,inplace=True)

df_train.head()

df_train.info()

df_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)

df_train.head()

df_train=pd.concat([df_train,Airline,Source,Destination],axis=1)

df_train.head()

df_train.info()

df_train.shape

"""# Test set"""

test_data=pd.read_excel('Test_set.xlsx')

print("Test data Info")

print(test_data.isnull().sum())
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

duration = list(test_data["Duration"])
for i in range(len(duration)):
  if('h' not in duration[i] or 'm' not in duration[i]):
    if('h' not in duration[i]):
      duration[i]='0h '+ duration[i]
    else:
      duration[i]=duration[i] + ' 0m'
duration_hour=[]
duration_min=[]

for i in range(len(duration)):
  time=list(duration[i].split(' '))
  duration_hour.append(int(time[0][:-1]))
  duration_min.append(int(time[1][:-1]))

test_data["Duration_hours"] = duration_hour
test_data["Duration_mins"] = duration_min
test_data.drop(["Duration"], axis = 1, inplace = True)

print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print("Shape of test data : ", data_test.shape)

data_test.head()

df_train.drop('Journey_Year',axis=1,inplace=True)

"""Feature Engineering
Finding out the best feature which will contribute and have good relation with target variable. Following are some of the feature selection methods,

heatmap

feature_importance_

SelectKBest


"""

df_train.shape

df_train.columns

##independent features
x=df_train.loc[:,['Total_Stops', 'Journey_Day', 'Journey_Month',
       'Dep_hour', 'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hour',
       'Duration_min', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]

x.head()

y=df_train['Price']

#finding co-relation between independent and dependent attributes using one hot encoding

plt.figure(figsize=(16,16))
sns.heatmap(df_train.corr(),annot=True, cmap = "RdYlGn")
#plt.show()

#FITTING MODEL USING RANDOM FOREST ALGORITHM
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

reg.score(x_train,y_train)

reg.score(x_test,y_test)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

metrics.r2_score(y_test, y_pred)

"""HYPERPARAMETER TURING"""

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2,random_state=42,n_jobs=1)

rf_random.fit(x_train,y_train)

rf_random.best_params_

prediction = rf_random.predict(x_test)

plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
#plt.show()

plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
#plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

import pickle
# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(reg, file)

model = open('flight_rf.pkl','rb')
forest = pickle.load(model)

y_prediction = forest.predict(x_test)

metrics.r2_score(y_test, y_prediction)



