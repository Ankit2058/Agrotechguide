#importing all the necessary library and frameworks
import numpy as np #used for number manipulation
import pandas as pd   #data manipulation library
import matplotlib.pyplot as plt  #library for data visualization
import seaborn as sns #statistical data visualization
from sklearn.preprocessing import LabelEncoder #for label encoding
from sklearn.model_selection import train_test_split #for splitting data
from sklearn.ensemble import RandomForestClassifier#random forest algorithm
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib  # Library to save and load trained models

# Load CSV file into a DataFrame
df = pd.read_csv(r'C:\Users\User\Desktop\Agrotech\agrotech.csv')

# Convert the labels into a Python list for machine learning
class_labels = df['label'].unique().tolist()

# Encode the labels into integers for machine learning compatibility
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
class_labels = le.classes_

# Divide dataset into input 'X' and output 'Y' 
X= df.drop('label', axis=1)
Y= df['label']

# Check if the trained model file exists
model_file = 'agrotech_model.pkl'
try:
    # Load the trained model from the file
    best_model = joblib.load(model_file)
except FileNotFoundError:
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, shuffle=True)
    
    # Train a RandomForestClassifier
    atg_model = RandomForestClassifier()
    atg_model.fit(X_train, Y_train)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Instantiate RandomizedSearchCV
    rscv_model = RandomizedSearchCV(estimator=atg_model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)

    # Fit RandomizedSearchCV to the training data
    rscv_model.fit(X_train, Y_train)

    # Get the best estimator
    best_model = rscv_model.best_estimator_

    # Save the trained model to a file
    joblib.dump(best_model, model_file)

# Taking user inputs
test_series = pd.Series(np.zeros(len(X.columns)), index=X.columns)
test_series['N'] = 90
test_series['P'] = 42
test_series['K'] = 43
test_series['temperature'] = 30
test_series['humidity'] = 60
test_series['ph'] = 6.5
test_series['rainfall'] = 31


# Convert test_series into a DataFrame with a single row
test_df = test_series.to_frame().T

# Predict the recommended crop
output = best_model.predict(test_df)[0]
recommended_crop = class_labels[output]
print("Recommended Crop:", recommended_crop)
