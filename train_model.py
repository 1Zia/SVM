import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ===============================
# Load dataset
# ===============================
data = pd.read_csv("Breast_Cancer.csv")

# Target column (confirmed)
target_col = "Status"

# Encode target
data[target_col] = data[target_col].astype("category").cat.codes

# Separate X & y
X = data.drop(target_col, axis=1)
y = data[target_col]

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# ✅ SAVE FEATURE NAMES
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train SVM
model = SVC(kernel="rbf", gamma=0.1, C=1)
model.fit(X_train, y_train)

# Save everything
pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(feature_names, open("features.pkl", "wb"))

print("✅ MODEL, SCALER & FEATURES SAVED")
