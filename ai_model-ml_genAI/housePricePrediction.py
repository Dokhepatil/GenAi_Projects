import pandas as pd
from sklearn.linear_model import LinearRegression


# df = pd.read_csv("housePrice.csv")
# print(df.head())

# model = LinearRegression()
# # train the  model
# model.fit(df[['area_sqr_ft']],df["price_lakhs"])

# # save this model to use
# # byt for now we r directly going to use it
# output = model.predict([[658]])
# print(output)

df = pd.read_csv("housePrice.csv")
df.head()

# Place this BEFORE model.fit()
df = df.dropna(subset=['price_lakhs'])
# Place this BEFORE model.fit()
df['price_lakhs'] = df['price_lakhs'].fillna(df['price_lakhs'].mean())


model = LinearRegression()
model.fit(df[['area_sqr_ft']], df['price_lakhs'])

output = model.predict([[1265]])
print(output)
