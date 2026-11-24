import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)

sales = pd.read_csv("used_car_sales.csv")
lots = pd.read_csv("auction_lots.csv")

sales["Accident"] = sales["Accident"].map({"yes": 1, "no": 0})
lots["Accident"] = lots["Accident"].map({"yes": 1, "no": 0})

print("used_car_sales.csv")
print(f"Rows: {sales.shape[0]}, Columns: {sales.shape[1]}")
print(sales.head())
print(f"Average Sale_Price: {sales['Sale_Price'].mean()}")

print()
print("auction_lots.csv")
print(f"Rows: {lots.shape[0]}, Columns: {lots.shape[1]}")
print(lots.head())

train, test = train_test_split(sales, test_size=0.30, random_state=42)
print()
print(f"Train rows: {len(train)}, Test rows: {len(test)}")

y_train = train["Sale_Price"]

features_a = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
X_train_a = sm.add_constant(train[features_a])
model_a = sm.OLS(y_train, X_train_a).fit()
print()
print("Model A")
print(model_a.summary())

features_b = ["Mileage_K", "Age_Years", "Horsepower", "Accident"]
X_train_b = sm.add_constant(train[features_b])
model_b = sm.OLS(y_train, X_train_b).fit()
print()
print("Model B")
print(model_b.summary())

print()
print(
    f"Adjusted R-squared  Model A: {model_a.rsquared_adj:.4f}"
    f"    Model B: {model_b.rsquared_adj:.4f}"
)

v1 = model_b
y_test = test["Sale_Price"]
baseline_avg = y_train.mean()
baseline_pred = np.full(len(test), baseline_avg)
v1_pred = v1.predict(sm.add_constant(test[features_b]))

baseline_r2 = r2_score(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
v1_r2 = r2_score(y_test, v1_pred)
v1_rmse = np.sqrt(mean_squared_error(y_test, v1_pred))

print()
print(f"Baseline  R-squared: {baseline_r2:.4f}   RMSE: ${baseline_rmse:,.2f}")
print(f"v1        R-squared: {v1_r2:.4f}   RMSE: ${v1_rmse:,.2f}")

low = min(y_test.min(), v1_pred.min(), baseline_avg)
high = max(y_test.max(), v1_pred.max(), baseline_avg)
pad = (high - low) * 0.05
low -= pad
high += pad

plt.figure(figsize=(6, 6))
plt.scatter(y_test, v1_pred, label="Test cars")
plt.plot([low, high], [low, high], linestyle="--", label="Predicted = actual")
plt.axhline(baseline_avg, linestyle="--", label="Baseline average")
plt.xlim(low, high)
plt.ylim(low, high)
plt.xlabel("Actual sale price ($)")
plt.ylabel("v1 predicted price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.close()


def decide(car, auction_price):
    row = pd.DataFrame([{name: car[name] for name in features_b}])
    X = sm.add_constant(row, has_constant="add")
    predicted = float(v1.predict(X[v1.model.exog_names]).iloc[0])
    if auction_price + 1500 < predicted:
        return "buy"
    return "pass"


def baseline_rule(auction_price):
    if auction_price + 1500 < baseline_avg:
        return "buy"
    return "pass"


def report_rule(name, decisions):
    bought = decisions == "buy"
    profit = (
        lots.loc[bought, "Later_Sale_Price"]
        - lots.loc[bought, "Auction_Price"]
        - 1500
    )
    print(name)
    print(f"Cars bought: {int(bought.sum())}")
    print(f"Bought cars that lose money: {int((profit < 0).sum())}")
    print(f"Total profit: ${profit.sum():,.2f}")


v1_decisions = lots.apply(
    lambda car: decide(car, car["Auction_Price"]), axis=1
)
baseline_decisions = lots["Auction_Price"].apply(baseline_rule)

print()
print(f"Train average Sale_Price: ${baseline_avg:,.2f}")
print()
report_rule("v1", v1_decisions)
print()
report_rule("Baseline", baseline_decisions)
