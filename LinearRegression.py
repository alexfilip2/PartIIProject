from ToolsStructural import *

# Sperate train and test data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

adjs, scores = load_regress_data(['NEO.NEOFAC_A'])
print(adjs.shape)
print(scores.shape)
X_train, X_test, y_train, y_test = train_test_split(adjs, scores, test_size=0.2, random_state=0)
model = LinearRegression()
# 2. Use fit
model.fit(X_train, y_train)
# 3. Check the score
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
print(y_pred)
print(mean_squared_error(y_pred, y_test))
