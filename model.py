import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================
# DIABETES MODEL
# ============================

diabetes = pd.read_csv("data/diabetes.csv")

X = diabetes.drop("Outcome", axis=1)
y = diabetes["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = RandomForestClassifier(n_estimators=200,random_state=42)

model.fit(x_train_scaled,y_train)

pred = model.predict(x_test_scaled)

print("Diabetes Accuracy:",accuracy_score(y_test,pred))

joblib.dump(model,"model/diabetes_model.pkl")
joblib.dump(scaler,"model/scaler.pkl")

print("Diabetes model saved")


# ============================
# HEART MODEL FINAL
# ============================

heart = pd.read_csv("data/heart.csv")

# Fix column names
heart.rename(columns={
'thalch':'thalach',
'num':'target'
}, inplace=True)

# Remove useless columns
heart = heart.drop(columns=['id','dataset'])

# Convert text to numbers
for col in heart.columns:
    if heart[col].dtype == "object":
        heart[col] = pd.factorize(heart[col])[0]

# Convert target to binary (IMPORTANT)
heart['target'] = heart['target'].apply(lambda x: 1 if x>0 else 0)

# Select required columns
heart = heart[
[
'age',
'sex',
'cp',
'trestbps',
'chol',
'fbs',
'restecg',
'thalach',
'exang',
'oldpeak',
'slope',
'ca',
'thal',
'target'
]
]

Xh = heart.drop("target",axis=1)
yh = heart["target"]

x_train,x_test,y_train,y_test = train_test_split(
Xh,yh,test_size=0.2,random_state=42
)

heart_scaler = StandardScaler()

x_train_scaled = heart_scaler.fit_transform(x_train)
x_test_scaled = heart_scaler.transform(x_test)

heart_model = RandomForestClassifier(n_estimators=300)

heart_model.fit(x_train_scaled,y_train)

pred = heart_model.predict(x_test_scaled)

print("Heart Accuracy:",accuracy_score(y_test,pred))

joblib.dump(heart_model,"model/heart_model.pkl")
joblib.dump(heart_scaler,"model/heart_scaler.pkl")

print("Heart model saved")


print("\nALL MODELS CREATED SUCCESSFULLY")