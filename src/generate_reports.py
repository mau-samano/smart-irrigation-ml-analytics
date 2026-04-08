import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

sns.set(style="whitegrid")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "raw", "train.csv")
reports_path = os.path.join(BASE_DIR, "reports")


df = pd.read_csv(data_path)

# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Irrigation_Need")
plt.title("Irrigation Need Distribution")

plt.savefig(os.path.join(reports_path, "target_distribution.png"))
plt.close()


# Soil Moisture
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Irrigation_Need", y="Soil_Moisture")
plt.title("Soil Moisture vs Irrigation Need")

plt.savefig(os.path.join(reports_path, "soil_moisture_vs_irrigation.png"))
plt.close()


# Temp
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Irrigation_Need", y="Temperature_C")
plt.title("Temperature vs Irrigation Need")

plt.savefig(os.path.join(reports_path, "temperature_vs_irrigation.png"))
plt.close()


# Humidity
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Irrigation_Need", y="Humidity")
plt.title("Humidity vs Irrigation Need")

plt.savefig(os.path.join(reports_path, "humidity_vs_irrigation.png"))
plt.close()

# Feature importance
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(["Irrigation_Need", "id"], axis=1)
y = df_encoded["Irrigation_Need"]

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importance.head(10).plot(kind="barh")
plt.title("Feature Importance")
plt.gca().invert_yaxis()

plt.savefig(os.path.join(reports_path, "feature_importance.png"))
plt.close()

print("Reports generated successfully in /reports")
