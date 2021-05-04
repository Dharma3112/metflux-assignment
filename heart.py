import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
plt.matshow(df.corr())
plt.show()
x_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.1,random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = Sequential()
model.add(Dense(13,activation='selu',kernel_regularizer=l2(0.001),input_shape=[13]))
model.add(Dense(26,activation='selu',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(13,activation='selu',kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,verbose=0)
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epoch = range(len(acc))

plt.figure(figsize=(16,8))

plt.subplot(1,2,1)
plt.plot(epoch,acc,label='acc')
plt.plot(epoch,val_acc,label='val_acc')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epoch,loss,label='loss')
plt.plot(epoch,val_loss,label='val_loss')
plt.title('Loss')
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), np.array(y_test).reshape(len(y_test),1)),1))