import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

data = pd.read_hdf('path/to/labeled_data.h5', key='df')

X = data.drop(columns=['label'])  # Features
y = data['label']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Assuming svm_model is the trained SVM model
joblib.dump(svm_model, 'svm_model.pkl')

# Load the saved SVM model
# svm_model = joblib.load('svm_model.pkl')


unseen_data = pd.read_csv('path/to/unseen_data.csv')







