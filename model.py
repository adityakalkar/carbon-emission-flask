import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import matplotlib.pyplot as plt


df = pd.read_csv('Carbon Emission.csv')

df['Recycling'] = df['Recycling'].apply(lambda x: ast.literal_eval(x))
df['Cooking_With'] = df['Cooking_With'].apply(lambda x: ast.literal_eval(x))
df['Body Type'] = df['Body Type'].apply(lambda x: convertBodyWeight(x))
df['Sex'] = df['Sex'].apply(lambda x: convertGender(x))
df['Diet'] = df['Diet'].apply(lambda x: convertDiet(x))
df['How Often Shower'] = df['How Often Shower'].apply(lambda x: convertShower(x))
df['Heating Energy Source'] = df['Heating Energy Source'].apply(lambda x: convertEnergy(x))
df['Transport'] = df['Transport'].apply(lambda x: convertTransport(x))
df['Vehicle Type'] = df['Vehicle Type'].apply(lambda x: convertVehicleEnergy(x))
df['Social Activity'] = df['Social Activity'].apply(lambda x: convertSocial(x))
df['Frequency of Traveling by Air'] = df['Frequency of Traveling by Air'].apply(lambda x: convertAir(x))
df['Waste Bag Size'] = df['Waste Bag Size'].apply(lambda x: convertTrash(x))
df['Energy efficiency'] = df['Energy efficiency'].apply(lambda x: convertEnergyEfficient(x))
df['Recycling'] = df['Recycling'].apply(lambda x: convertRecycling(x))
df['Cooking_With'] = df['Cooking_With'].apply(lambda x: convertCooking(x))

x = df.drop(['CarbonEmission'], axis=1)
y = df['CarbonEmission']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

params = {
    'learning_rate': 0.17425361703048103,
    'max_depth': 3,
    'max_features': 'sqrt',
    'min_samples_leaf': 10,
    'min_samples_split': 20,
    'n_estimators': 500
}

gb_reg = GradientBoostingRegressor(**params, random_state=42)

gb_reg.fit(xtrain, ytrain)

pred = gb_reg.predict(xtest)

mse = mean_squared_error(ytest, pred)
r2 = r2_score(ytest, pred)
mae = mean_absolute_error(ytest, pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

joblib.dump(gb_reg, 'gradient_boosting_model.pkl')

# Plot prediction vs. actual
plt.scatter(ytest, pred, alpha=0.5)
plt.plot([min(ytest), max(ytest)], [min(ytest), max(ytest)], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Prediction vs. Actual')
plt.legend()
plt.show()
