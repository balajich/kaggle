# Artificial neural networks
# Multilayer perceptrons

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

seed = 7
train_data = pd.read_csv("./kaggle/input/titanic/train.csv")
train_data.head()

pred_data = pd.read_csv("./kaggle/input/titanic/test.csv")
pred_data.head()
y_train_complete = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train_complete = pd.get_dummies(train_data[features])
X_pred = pd.get_dummies(pred_data[features])

# split into 80% for train and 20% for test
X_train, X_test, y_train, y_test = train_test_split(X_train_complete, y_train_complete, test_size=0.2,
                                                    random_state=seed)

# create model
model = Sequential()
model.add(Dense(6, input_dim=5, activation="relu", kernel_initializer="normal"))
model.add(Dense(3, activation="relu", kernel_initializer="normal"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="normal"))

# Compile model
# binary_crossentropy = logarithmic loss
# adam = gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=200, batch_size=10)

# Evaluating model with the training data
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print('loss: ', scores[0])
y_test_prob=model.predict(X_test, batch_size=10)
y_pred = []
for y in y_test_prob:
    if y > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print('Confusion Matrix :', confusion_matrix(y_test, y_pred))
print('Accuracy Score :', accuracy_score(y_test, y_pred))
print('Classification Report :')
print(classification_report(y_test, y_pred))


y_pred_porb = model.predict(X_pred, batch_size=10)
y_pred = []
for y in y_pred_porb:
    if y > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
output = pd.DataFrame({'PassengerId': pred_data.PassengerId, 'Survived': y_pred})
output.to_csv('my_submission.csv', index=False)
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import accuracy_score
#
# print('Confusion Matrix :', confusion_matrix(y_test, y_pred))
# print('Accuracy Score :', accuracy_score(y_test, y_pred))
# print('Classification Report :')
# print(classification_report(y_test, y_pred))

# output = pd.DataFrame({'PassengerId': pred_data.PassengerId, 'Survived': y_pred})
# output.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")
