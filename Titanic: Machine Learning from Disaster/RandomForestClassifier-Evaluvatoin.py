import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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

# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X_train_complete, y_train_complete, test_size=0.2,
                                                    random_state=seed)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

print('Confusion Matrix :', confusion_matrix(y_test, y_pred))
print('Accuracy Score :', accuracy_score(y_test, y_pred))
print('Classification Report :')
print(classification_report(y_test, y_pred))

# output = pd.DataFrame({'PassengerId': pred_data.PassengerId, 'Survived': y_pred})
# output.to_csv('my_submission.csv', index=False)
# print("Your submission was successfully saved!")
