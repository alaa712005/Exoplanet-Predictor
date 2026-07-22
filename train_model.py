import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# ==========================
# 1. تحميل البيانات
# ==========================

train_df = pd.read_csv("train.csv")

ID_COL = "star_id"
TARGET_COL = "label"

X = train_df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
y = train_df[TARGET_COL]

# ==========================
# 2. تحديد أنواع الأعمدة
# ==========================

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# ==========================
# 3. تجهيز الـ Preprocessing
# ==========================

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# ==========================
# 4. إنشاء Pipeline
# ==========================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

# ==========================
# 5. تدريب النموذج
# ==========================

print("جاري تدريب النموذج...")

pipeline.fit(X, y)

print("\n==============================")
print("Columns used for training:")
print(X.columns.tolist())
print("==============================\n")

# ==========================
# 6. حفظ النموذج
# ==========================

model_data = {
    "pipeline": pipeline,
    "num_cols": num_cols,
    "cat_cols": cat_cols
}

joblib.dump(model_data, "exoplanet_model.pkl")

print("✅ تم حفظ النموذج بنجاح في ملف exoplanet_model.pkl")