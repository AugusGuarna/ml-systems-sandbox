from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib 

X,y = load_iris(return_X_y=True)
model = LogisticRegression(random_state=7916).fit(X,y)

joblib.dump(model,'models/v1.pkl')