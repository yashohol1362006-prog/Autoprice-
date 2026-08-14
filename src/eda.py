import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib 

df = pd.read_csv(r"C:\Users\Yash Ohol\OneDrive\Autoprice\data\cars_dataset.csv")
print('Your first 5 rows are ')
print(df.head())

print('\n SHAPE OF DATASET :')
print(df.shape)
print('\nINFORMATION :')
print(df.info())
print('\nSTATISTICAL SUMMARY :')
print(df.describe())
print('\nMISSING VALUES ARE :')
print(df.isna().sum())
print('\n DUPLICATE :')
print('Duplicate Rows :', df.duplicated().sum())

# Checking Duplicate Rows
print('\n Duplicate rows :')
print(df[df.duplicated()].head())

# Checking Unique values
print('\n UNIQUE VALUES :')
print(df.nunique())

# Checking Categorial values
# transmisssion, fuelType, make values, model count

print('\n TRANSMISSION VALUES')
print(df['transmission'].value_counts())

print('\n FUEL TYPES ARE :')
print(df['fuelType'].value_counts())

print('\n MAKE VALUES :')
print(df['Make'].value_counts())

print('\n MODEL COUNT :')
print(df['model'].nunique())

print('PRICE DISTRIBUTION :')
sns.histplot(x = 'price', data= df, bins= 100, kde= True)
plt.show()

plt.figure(figsize=(8, 5))

sns.scatterplot(x= 'mileage', y= 'price', data= df, )
plt.title('Mileage VS Price ')
plt.xlabel('Mileage')
plt.ylabel('Price')

plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x= 'year', y= 'price', data= df)
plt.title('Year VS Price')
plt.xlabel('Year')
plt.ylabel('Price')

plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x= 'engineSize', y= 'price', data= df)
plt.title('Engine Size VS Price')
plt.xlabel('Engine Size')
plt.ylabel('Price')

plt.show()


print('n CORRELATION MATRIX')

correlation = df[['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']].corr()
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot= True, cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x= 'transmission', y= 'price', data= df)
plt.title('Transmission VS Price')
plt.xlabel('Transmission')
plt.ylabel('price')

plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x= 'fuelType', y= 'price', data= df)
plt.title('Fuel Typee VS Price')
plt.xlabel('fuelType')
plt.ylabel('Price')

plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x= 'Make', y= 'price', data= df)
plt.title('Car Make VS Price')
plt.xlabel('Make')
plt.ylabel('Price')

plt.show()

print('n OUTLIER CHECK')
plt.figure(figsize=(8, 5))
sns.boxplot(x = 'price', data= df)
plt.title('Price Outliers')
plt.xlabel('Price Outliers')
plt.xlabel('Price')

plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='mileage', data= df)
plt.title('Mieage Outliers')
plt.xlabel('Mileage')

plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x= 'year', data= df)
plt.title('Year Outliers')
plt.xlabel('Year')

plt.show()

print('\n DUPLICATE ROW COUNT :')
print(df.duplicated().sum())

print('n DUPLICATE EXAMPLES :')
print(df[df.duplicated(keep=False)].sort_values(by= ['model', 'year', 'price']).head(20)) 


print(df['transmission'].value_counts())
print(df['fuelType'].value_counts())
print(df['Make'].value_counts())

# We are splitting Categorial values into each now Feature/Column
df_cleaned = pd.get_dummies(df, columns= ['transmission', 'fuelType', 'Make', 'model'])

# Here we Have gotten the splitted cateforial values into Trrue OR False so now
# we are converting it into numerical values then it will be useful for our prediction 
print(df_cleaned.astype(int))

# Now we'll select feature to put on X and y axis 
x = df_cleaned.drop(columns=['price'])
y = df_cleaned['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42 )

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "models/linear_regression_model.pkl")
joblib.dump(x.columns.tolist(), "models/model_columns.pkl")
y_pred = model.predict(X_test)
print('\n',y_pred)

r2 = r2_score(y_test, y_pred)
print(f'\nOur R2 value is : {r2}')

n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n-p-1))
print(f'\nOur Adjusted r2 is : {adjusted_r2}') 

mae = mean_absolute_error(y_test, y_pred)
print(f'\nThe MAE (Mean Absolute Error) of our dataset is : {mae} ')

mse = mean_squared_error(y_test, y_pred)
print(f'\n The Mean Squared Eroor is : {mse}')

rmse = mse ** 0.5
print(f"]\n Our RMSE is : {rmse}")

# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)

coefficient = pd.DataFrame({'Feature':x.columns, 'Coefficients':model.coef_})
print(coefficient)
print("\nMost Positive Coefficients:")
print(
    coefficient.sort_values(
        by='Coefficients',
        ascending=False
    ).head(10)
)

print("\nMost Negative Coefficients:")
print(
    coefficient.sort_values(
        by='Coefficients',
        ascending=True
    ).head(10)
)
comperison = pd.DataFrame({'Actual Price' : y_test, 'Predicted Price': y_pred})
print(comperison.head(10))
comperison['Error'] = comperison['Actual Price'] - comperison['Predicted Price']
print(comperison.head(10))

sns.scatterplot( x = y_test, y = y_pred )
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.title('Actual Price VS Predicted Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()
# Residuals on Graph
residuals = y_test - y_pred
sns.scatterplot(x= y_pred, y =y_test)
plt.axhline(y = 0)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals VS Predicted Price')
plt.show()

sns.histplot(residuals, kde =True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')

plt.show()

# Worst Predictions 
comperison['Absolute Error'] = abs(comperison['Actual Price'] - comperison['Predicted Price'])
worst_prediction = comperison.sort_values(by= 'Absolute Error', ascending=False)
print(worst_prediction.head(10))

residuals = y_test - y_pred

sns.histplot(x = residuals, kde= True)

plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')

plt.show()


new_car = pd.DataFrame({
    'year': [2019],
    'mileage': [30000],
    'tax': [145],
    'mpg': [50.0],
    'engineSize': [1.6],
    'transmission': ['Manual'],
    'fuelType': ['Petrol'],
    'Make': ['BMW'],
    'model': ['3 Series']
})
new_car_encoded=  pd.get_dummies(new_car, columns= ['model', 'transmission','fuelType', 'Make'])

new_car_encoded = new_car_encoded.reindex(
    columns=x.columns,
    fill_value=0
)
prediction = model.predict(new_car_encoded)
print('\nPrediction Car : ', prediction[0])

