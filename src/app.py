import pandas as pd
import numpy as np  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#importing the CSV here
df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')
 
#convert category sex and smoker to num (need for calculate correlation coeficient)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0) 

#convert the getion category to num 
def regtoNum (x):
    if x == 'southwest':
        return 1
    elif x == 'southeast':
        return 2
    elif x == 'northwest':
        return 3
    elif x == 'northeast':   
        return 4
    else:
        return 'region sin determinar'

df['region'] = df['region'].apply(lambda x: regtoNum(x)) 

X = df.drop(['charges'], axis=1)
y = df['charges']

#creo los dataframe de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, y)

edad = 33
sex = 1
bmi = 22
children = 0
smoker = 1
region= 3

#predigo el target (charges) para los valores seteados
print('Predicted prima : \n', regr.predict([[edad,sex,bmi,children,smoker,region]]))