import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


df = pd.read_csv('FuelConsumption.csv')
c_df = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

plt.scatter(c_df['ENGINESIZE'], c_df['CO2EMISSIONS'])
plt.title('Scatter plot of Enginesize and CO2 Emission')
plt.xlabel('ENGINE SIZE')
plt.ylabel('CO2 EMISSIONS')
plt.show()

def train_test(df0):
    msk = np.random.rand(len(df0)) < 0.8
    train0 = c_df[msk]
    test0 = c_df[~msk]
    return train0, test0

train, test = train_test(df)

model = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE']])
y = np.asanyarray(train[['CO2EMISSIONS']])
model.fit(x, y)

test_y = np.asanyarray(test[['CO2EMISSIONS']])  # Score of prediction

inputVal = float(input('Enter a EngineSize value for testing the model from Test Data'))
test_x = np.asanyarray(test[['ENGINESIZE']])

val = np.array(inputVal)

new_val = val.reshape(1, 1)

predicted_val = model.predict(new_val)
print('Predicted value for '+str(new_val.item())+' is : '+str(predicted_val.item()))
test_y_ = model.predict(test_x)

from sklearn.metrics import r2_score

score = r2_score(test_y_, test_y)

plt.scatter(train['ENGINESIZE'],train['CO2EMISSIONS'], color= 'green')
plt.title('Graph of Linear Regression')
plt.xlabel('ENGINE SIZE')
plt.ylabel('CO2 EMISSIONS')
plt.plot(test['ENGINESIZE'], test_y_, color='red')
plt.show()
print('Prediction Accuracy: '+str(score))
