
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)


url = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)
df = pd.read_table(url, header=None, names=["label", "message"])
df.head()


print("Shape:", df.shape)
print(df["label"].value_counts())
sns.countplot(x="label", data=df)
plt.title("Class distribution")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)


tfidf = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)


y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


RocCurveDisplay.from_estimator(clf, X_test_vec, y_test)
plt.title("ROC Curve")
plt.show()


cv_scores = cross_val_score(
    clf, tfidf.transform(df["message"]), df["label"], cv=5, scoring="accuracy"
)
print("CV accuracy scores:", cv_scores)
print("CV mean accuracy:", cv_scores.mean().round(4))

sample = [
    "Congratulations! You've won a free cruise to the Bahamas. Call now!",
    "Hey, are we still on for lunch tomorrow?"
]
sample_vec = tfidf.transform(sample)
preds = clf.predict(sample_vec)
for sms, p in zip(sample, preds):
    print(f"> {sms}\n→ Prediction: {p}\n")


import joblib
joblib.dump(clf, "spam_logreg_model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
print("Artifacts saved to disk.")
