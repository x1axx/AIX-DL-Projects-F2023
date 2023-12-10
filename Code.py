import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
#import data
data = pd.read_csv('wines_SPA.csv')
# dataset description and preprocessing
data.head()
data.info()
#date preprocessing
#delete the rows with missing values and the column country
data['year'].replace('N.V.', np.nan, inplace=True)
data.dropna(inplace=True)
data.drop(columns=['country'], inplace=True)
#check the infomation of the dataset
data.info()
data.describe()
## dataset description
#check the information of the winery
data['winery'].value_counts()
#check the information of the name
data['wine'].value_counts()
#check the distribution of the year
data['year'] = data['year'].astype('int64')
YearCategory = pd.cut(data['year'], 
                        bins=[-float('inf'), 2000, 2005, 2010, 2015, 2020, float('inf')],
                        labels=['<2000', '2000-2004', '2005-2009', '2010-2014', '2015-2019', '2020+'])
print(YearCategory.value_counts())
#plot the distribution of the year
plt.figure(figsize=(5, 3))
sns.countplot(YearCategory)
plt.title('Year Category')
plt.show()
#check the distribution of the rating
rating = data['rating'].value_counts()
print(rating)

plt.figure(figsize=(5, 3))
plt.hist(data['rating'], bins=20)
plt.title('Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
#check the distribution of the price
PriceCategory = pd.cut(data['price'], 
                        bins=[-float('inf'), 50, 100, 200, 300, 400, 500, float('inf')],
                        labels=['<50', '50-100', '100-200', '200-300', '300-400', '400-500', '500+'])
print(PriceCategory.value_counts())

sns.countplot(PriceCategory)
plt.title('Price Category')
plt.ylabel('Price(EUR)')
#check the relationship between price and rating
plt.figure(figsize=(5, 3))
sns.scatterplot(x='price', y='rating', data=data)
plt.title('Price and Rating')
plt.show()
#check the average price of each rating
price_rating = data.groupby('rating')['price'].mean()
print(price_rating)

plt.figure(figsize=(5, 3))
sns.barplot(x=price_rating.index, y=price_rating.values)
plt.title('Price Rating')
plt.xlabel('Rating')
plt.ylabel('Price(EUR)')
plt.show()
#check the distribution of the type
print(data['type'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(data['type'])
plt.title('Type')
plt.show()
#calculate the mean price of each type
mean_price = data.groupby('type')['price'].mean()
print(mean_price)
#plot the mean price of each type
plt.figure(figsize=(6, 4))
sns.barplot(x=mean_price.index, y=mean_price.values)
plt.title('Mean Price')
plt.xlabel('Type')
plt.xticks(rotation=90)
plt.ylabel('Price(EUR)')
plt.show()
#check the distribution of the body
print(data['body'].value_counts())
#check the distribution of the acidity
print(data['acidity'].value_counts())
### data scaling
#encoding the dataset
from sklearn.preprocessing import LabelEncoder
lable = LabelEncoder()
data['winery'] = lable.fit_transform(data['winery'])
data['wine'] = lable.fit_transform(data['wine'])
data['region'] = lable.fit_transform(data['region'])
data['type'] = lable.fit_transform(data['type'])

data.head()
#data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.head()
#check the seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True)
plt.show()
# train and evaluate the model
#split the dataset, target is price
x_train, x_test, y_train, y_test = train_test_split(data_scaled.drop(columns=['price']), data_scaled['price'], test_size=0.2, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
### define the evaluation function and plot function
#defining evaluation function
def evaluate(y_test, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    print('MSE: ', MSE)
    print('MAE: ', MAE)
    print('R2: ', R2)
    Result = [MSE, MAE, R2]
    return Result
#defining plot function
def plot_result(y_test, y_pred):
    fig, ax = plt.subplots(figsize = (12,4))
    idx = np.asarray([i for i in range(50)])
    width = 0.2
    ax.bar(idx, y_test.values[:50], width = width)
    ax.bar(idx+width, y_pred[:50], width = width)
    ax.legend(["Actual", "Predicted"])
    ax.set_xticks(idx)
    ax.set_xlabel("Index")
    ax.set_ylabel("Price")
    fig.tight_layout()
    plt.show()
    plt.scatter(y_test, y_pred, color='red')
    plt.plot(y_test, y_test, color='blue', linewidth=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.show()
### train the model
#use linear regression train the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

#evaluate the model
LR = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#use polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
lr_poly = LinearRegression()
lr_poly.fit(x_train_poly, y_train)
y_pred_poly = lr_poly.predict(x_test_poly)

#evaluate the model
Poly = evaluate(y_test, y_pred_poly)
#plot the results
plot_result(y_test, y_pred_poly)
#use random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

#evaluate the model
RF = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#use decision tree regression
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

#evaluate the model
DT = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#use KNN regression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

#evaluate the model
KNN = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#compare the results
pd.DataFrame([LR, Poly, RF, DT, KNN], columns=['MSE', 'MAE', 'R2'], index=['Linear Regression', 'Polynomial Regression', 'Random Forest', 'Decision Tree', 'KNN'])
### rating as target
#split the dataset, target is rating
x_train, x_test, y_train, y_test = train_test_split(data_scaled.drop(columns=['rating']), data_scaled['rating'], test_size=0.2, random_state=0)
#defining plot function
def plot_result(y_test, y_pred):
    fig, ax = plt.subplots(figsize = (12,4))
    idx = np.asarray([i for i in range(50)])
    width = 0.2
    ax.bar(idx, y_test.values[:50], width = width)
    ax.bar(idx+width, y_pred[:50], width = width)
    ax.legend(["Actual", "Predicted"])
    ax.set_xticks(idx)
    ax.set_xlabel("Index")
    ax.set_ylabel("rating")
    fig.tight_layout()
    plt.show()
#use linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

#evaluate the model
LR = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#use polynomial regression
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
lr_poly = LinearRegression()
lr_poly.fit(x_train_poly, y_train)
y_pred_poly = lr_poly.predict(x_test_poly)

#evaluate the model
Poly = evaluate(y_test, y_pred_poly)
#plot the results
plot_result(y_test, y_pred_poly)
#use random forest regression
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

#evaluate the model
RF = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#use decision tree regression
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

#evaluate the model
DT = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#use KNN regression
knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

#evaluate the model
KNN = evaluate(y_test, y_pred)
#plot the results
plot_result(y_test, y_pred)
#compare the results
pd.DataFrame([LR, Poly, RF, DT, KNN], columns=['MSE', 'MAE', 'R2'], index=['Linear Regression', 'Polynomial Regression', 'Random Forest', 'Decision Tree', 'KNN'])