from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


y = y.astype(int)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
model.fit(X_train, y_train)
print("New accuracy:", model.score(X_test, y_test))

joblib.dump(model, "digital_model.pkl")

digit = X.iloc[0].values.reshape(28, 28)
# print(model.score(X_test, y_test))

# y_pred = model.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(cm)

# disp.plot()
# plt.show()

plt.imshow(digit, cmap="gray")
# plt.title(f"Label: {y.iloc[0]}")
plt.title(f"Predicted: {model.predict(X_test[:1])[0]}")
plt.axis("off")
plt.show()