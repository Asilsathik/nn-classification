# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement
<h4>

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

</h4>

## Neural Network Model
![image](https://github.com/Praveen0500/nn-classification/assets/120218611/aee7bc70-25ea-44bd-b16f-437b96ee52b2)



## DESIGN STEPS

### STEP 1:
Import the necessary packages & modules
### STEP 2:
Load and read the dataset
### STEP 3:
Perform pre processing and clean the dataset
### STEP 4:
Encode categorical value into numerical values using ordinal/label/one hot encoding
### STEP 5:
Visualize the data using different plots in seaborn
### STEP 6:
Normalize the values and split the values for x and y
### STEP 7:
Build the deep learning model with appropriate layers and depth
### STEP 8:
Analyze the model using different metrics
### STEP 9:  
Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration
### STEP 10:
Save the model using pickle
### STEP 11:
Using the DL model predict for some random inputs


## PROGRAM

### Name: MOHAMED ASIL M
### Register Number: 212222230080

```py


import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
```




```py
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
```


```py
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
```

```py
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
```

```py
import numpy as np
import matplotlib.pylab as plt
```

```py
customer_df = pd.read_csv('customers.csv')
```

```py
customer_df.columns
customer_df.dtypes
customer_df.shape
```

```py
customer_df.isnull().sum()
```

```py
customer_df_cleaned = customer_df.dropna(axis=0)
```

```py
customer_df_cleaned.isnull().sum()
customer_df_cleaned.shape
customer_df_cleaned.dtypes
```
```py
customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()
customer_df_cleaned['Profession'].unique()
customer_df_cleaned['Spending_Score'].unique()
customer_df_cleaned['Var_1'].unique()
customer_df_cleaned['Segmentation'].unique()
```

```py
categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)
```
```py
customers_1 = customer_df_cleaned.copy()
```

```py
customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])


```

```py
customers_1.dtypes
```
```py
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes
```
```py
customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
```

```py
# Calculate the correlation matrix
corr = customers_1.corr()

# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
```
```py
sns.pairplot(customers_1)

```
```py

sns.distplot(customers_1['Age'])
plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])
```
```py
plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=customers_1)
```

```py
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=customers_1)
```
```py
plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=customers_1)
```

```py
customers_1.describe()
customers_1['Segmentation'].unique()
```

```py
X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values
y1 = customers_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
```
```py
y1.shape
```
```py
y = one_hot_enc.transform(y1).toarray()
y.shape
```
```py
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
```
```py
X_train[0]
X_train.shape
```
```py

scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
```
```py
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
```
```py
# Creating the model
ai_brain = Sequential([
  Dense(4,input_shape=(8,)),
  Dense(8,activation='relu'),
  Dense(16,activation='relu'),
  Dense(4,activation='softmax'),
])
```
```py
ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
```
```py
early_stop = EarlyStopping(monitor='val_loss', patience=2)
ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,
             batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
```
```py
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
```
```py
metrics[['loss','val_loss']].plot()
```
```py
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
```
```py
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
```
```py
# Saving the Model
ai_brain.save('customer_classification_model.h5')
# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

# Loading the Model
ai_brain = load_model('customer_classification_model.h5')
# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

```

```py
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```





## Dataset Information
![image](https://github.com/Asilsathik/nn-classification/assets/119476247/24d17b33-af64-4f8f-a601-95c89d36cac9)


## OUTPUT

### Heat map

![image](https://github.com/Asilsathik/nn-classification/assets/119476247/23c4c243-2670-4da4-9ba1-2cb8eef90d52)


### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Asilsathik/nn-classification/assets/119476247/b410bc2a-582e-45f5-87e4-f9576ec5f1af)



### Classification Report

![image](https://github.com/Asilsathik/nn-classification/assets/119476247/30cfefb0-1545-4fb9-99ea-bd3adab2ced4)




### Confusion Matrix

![image](https://github.com/Asilsathik/nn-classification/assets/119476247/9534229c-2d7c-41b9-892c-c109717d1efa)

### New Sample Data Prediction

![image](https://github.com/Asilsathik/nn-classification/assets/119476247/f8091fb0-476d-4b1a-b525-ac8872faedb9)


## RESULT
### A neural network classification model is developed for the given dataset.
