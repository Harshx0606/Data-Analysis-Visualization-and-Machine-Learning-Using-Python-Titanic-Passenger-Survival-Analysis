"""
Mini Project: Data Analysis, Visualization and Machine Learning Using Python
Dataset : Titanic passenger dataset (891 passengers, real-world data)
Name    : Harsh Choudhary
Roll No : 25SCS1003001231
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sns.set_theme(style="whitegrid")
OUT = "figs"
os.makedirs(OUT, exist_ok=True)
results = {}

# ---------------------------------------------------------------- 1. LOAD
df = sns.load_dataset("titanic")
results["shape"] = list(df.shape)
results["missing"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
results["survival_rate"] = round(df["survived"].mean() * 100, 1)

# ---------------------------------------------------------------- 2. PREPROCESSING
df["age"] = df["age"].fillna(df.groupby(["pclass", "sex"])["age"].transform("median"))
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df = df.drop(columns=["deck", "embark_town", "alive", "class", "who", "adult_male", "alone"])
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)
df["age_group"] = pd.cut(df["age"], [0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
results["missing_after"] = int(df[["age", "embarked"]].isna().sum().sum())

# ---------------------------------------------------------------- 3. ANALYSIS (NumPy / Pandas)
results["desc"] = df[["age", "fare", "family_size"]].describe().round(2).to_dict()
results["by_sex"] = (df.groupby("sex")["survived"].mean() * 100).round(1).to_dict()
results["by_class"] = (df.groupby("pclass")["survived"].mean() * 100).round(1).to_dict()
results["by_embarked"] = (df.groupby("embarked")["survived"].mean() * 100).round(1).to_dict()
results["by_agegroup"] = (df.groupby("age_group", observed=True)["survived"].mean() * 100).round(1).to_dict()
results["by_alone"] = (df.groupby("is_alone")["survived"].mean() * 100).round(1).to_dict()
results["fare_median_by_class"] = df.groupby("pclass")["fare"].median().round(2).to_dict()
results["age_mean"] = round(float(np.mean(df["age"])), 2)
results["age_std"] = round(float(np.std(df["age"])), 2)

# ---------------------------------------------------------------- 4. VISUALISATION
# Fig 1: Survival count
fig, ax = plt.subplots(figsize=(5, 3.6))
sns.countplot(x="survived", data=df, palette=["#d9534f", "#5cb85c"], ax=ax)
ax.set_xticklabels(["Did not survive", "Survived"])
ax.set_xlabel("")
ax.set_title("Overall Survival Count")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_survival.png", dpi=150); plt.close()

# Fig 2: Survival by sex and class
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
sns.barplot(x="sex", y="survived", data=df, palette="Set2", ax=axes[0], errorbar=None)
axes[0].set_title("Survival Rate by Sex"); axes[0].set_ylabel("Survival rate")
sns.barplot(x="pclass", y="survived", hue="sex", data=df, palette="Set1", ax=axes[1], errorbar=None)
axes[1].set_title("Survival Rate by Class and Sex"); axes[1].set_ylabel("")
axes[1].set_xlabel("Passenger class")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_sex_class.png", dpi=150); plt.close()

# Fig 3: Age distribution
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
sns.histplot(df["age"], bins=30, kde=True, color="#337ab7", ax=axes[0])
axes[0].set_title("Age Distribution")
sns.kdeplot(data=df, x="age", hue="survived", fill=True, common_norm=False,
            palette=["#d9534f", "#5cb85c"], ax=axes[1])
axes[1].set_title("Age Density by Survival")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_age.png", dpi=150); plt.close()

# Fig 4: Fare boxplot by class
fig, ax = plt.subplots(figsize=(5.5, 3.8))
sns.boxplot(x="pclass", y="fare", data=df, palette="pastel", ax=ax)
ax.set_ylim(0, 300); ax.set_title("Fare by Passenger Class"); ax.set_xlabel("Passenger class")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_fare.png", dpi=150); plt.close()

# Fig 5: Age group and family size
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
sns.barplot(x="age_group", y="survived", data=df, palette="viridis", ax=axes[0], errorbar=None)
axes[0].set_title("Survival Rate by Age Group"); axes[0].set_xlabel(""); axes[0].set_ylabel("Survival rate")
axes[0].tick_params(axis="x", rotation=20)
sns.barplot(x="family_size", y="survived", data=df, palette="magma", ax=axes[1], errorbar=None)
axes[1].set_title("Survival Rate by Family Size"); axes[1].set_ylabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_age_family.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 5. ENCODING + CORRELATION
ml = df.copy()
ml["sex"] = ml["sex"].map({"male": 0, "female": 1})
ml = pd.get_dummies(ml, columns=["embarked"], drop_first=True, dtype=int)
ml = ml.drop(columns=["age_group"])

fig, ax = plt.subplots(figsize=(7, 5.2))
sns.heatmap(ml.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_corr.png", dpi=150); plt.close()
corr = ml.corr()["survived"].drop("survived").sort_values()
results["corr_survived"] = corr.round(3).to_dict()

# ---------------------------------------------------------------- 6. MACHINE LEARNING
X = ml.drop(columns=["survived"])
y = ml["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
results["split"] = [len(X_train), len(X_test)]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}
results["models"] = {}
cms = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    results["models"][name] = {
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision": round(precision_score(y_test, pred) * 100, 2),
        "recall": round(recall_score(y_test, pred) * 100, 2),
        "f1": round(f1_score(y_test, pred) * 100, 2),
    }
    cms[name] = confusion_matrix(y_test, pred)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_confusion.png", dpi=150); plt.close()

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
results["importance"] = imp.sort_values(ascending=False).round(3).head(6).to_dict()
fig, ax = plt.subplots(figsize=(6, 4))
imp.plot(kind="barh", color="#337ab7", ax=ax)
ax.set_title("Random Forest Feature Importance")
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_importance.png", dpi=150); plt.close()

with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
# ---------------------------------------------------------------- 7. OUTPUT SUMMARY
print("=" * 60)
print("TITANIC MINI PROJECT - OUTPUT")
print("Name: Harsh Choudhary | Roll No: 25SCS1003001231")
print("=" * 60)
print("Dataset shape            :", results["shape"])
print("Overall survival rate    : %s %%" % results["survival_rate"])
print("Missing values (before)  :", results["missing"])
print("Missing values (after)   :", results["missing_after"])
print("\n--- Descriptive statistics ---")
print(pd.DataFrame(results["desc"]).to_string())
print("\n--- Survival rate (%) ---")
print("By sex        :", results["by_sex"])
print("By class      :", results["by_class"])
print("By embarked   :", results["by_embarked"])
print("By age group  :", results["by_agegroup"])
print("Alone (1/0)   :", results["by_alone"])
print("\n--- Correlation of features with survival ---")
for k, v in results["corr_survived"].items():
    print("  %-12s %7.3f" % (k, v))
print("\n--- Machine learning ---")
print("Train / Test samples :", results["split"])
print("%-22s %9s %10s %8s %6s" % ("Model", "Accuracy", "Precision", "Recall", "F1"))
for k, v in results["models"].items():
    print("%-22s %8.2f%% %9.2f%% %7.2f%% %5.2f%%" % (k, v["accuracy"], v["precision"], v["recall"], v["f1"]))
for name, cm in cms.items():
    print("\nConfusion matrix - %s\n%s" % (name, cm))
print("\nTop features (Random Forest):", results["importance"])
print("\nGraphs saved in folder: figs/")
