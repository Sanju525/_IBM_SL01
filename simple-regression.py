import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

st.title('Machine Learning Alogorithms')

st.header('1. Simple Linear Regression')

df = pd.read_csv('FuelConsumption.csv')
c_df = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

add_selectbox = st.sidebar.selectbox(
    'Show Raw Data-sets',
    ('None', \
     'Full Raw Data', \
     'Scatter Plot Data', \
     'Train Data', \
     'Test Data')
)
if add_selectbox == 'Full Raw Data':
    st.subheader('Full Raw Data')
    st.dataframe(df)
elif add_selectbox == 'Scatter Plot Data':
    st.subheader('Scatter Plot Data')
    st.dataframe(c_df)

plt.scatter(c_df['ENGINESIZE'], c_df['CO2EMISSIONS'])
plt.title('Scatter plot of Enginesize and CO2 Emission')
plt.xlabel('ENGINE SIZE')
plt.ylabel('CO2 EMISSIONS')
# plt.show()
st.pyplot()


@st.cache
def train_test(df0):
    msk = np.random.rand(len(df0)) < 0.8
    train0 = c_df[msk]
    test0 = c_df[~msk]
    return train0, test0


train, test = train_test(df)
if add_selectbox == 'Train Data':
    st.subheader('Training Data')
    st.dataframe(train)
elif add_selectbox == 'Test Data':
    st.subheader('Test Data')
    st.write('Note: This Data is Randomly Picked ')
    st.dataframe(test)

model = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE']])
y = np.asanyarray(train[['CO2EMISSIONS']])
model.fit(x, y)

test_y = np.asanyarray(test[['CO2EMISSIONS']])  # Score of prediction

inputVal = st.number_input('Enter a EngineSize value for testing the model from Test Data[ See Sidebar ]')
test_x = np.asanyarray(test[['ENGINESIZE']])

val = np.array(inputVal)

new_val = val.reshape(1, 1)

predicted_val = model.predict(new_val)
st.write('Predicted value for ' + str(new_val.item()) + ' is : ' + str(predicted_val.item()))
test_y_ = model.predict(test_x)

from sklearn.metrics import r2_score

score = r2_score(test_y_, test_y)
st.subheader('The Prediction Accuracy of this model is ' + str(score))

st.header('')
st.subheader('Green = ScatterPlot')
st.subheader('Red = Prediction for Test cases')
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='green')
plt.title('Graph of Linear Regression')
plt.xlabel('ENGINE SIZE')
plt.ylabel('CO2 EMISSIONS')
plt.plot(test['ENGINESIZE'], test_y_, color='red')
st.pyplot()

st.info("Hello, check the box below to show the model code!")


@st.cache
def codeData():
    file_object = open('modelC.py,'r')
    data00 = file_object.read()
    return data00


if st.checkbox('Show Model Code'):
    data = codeData()
    st.code(data)
