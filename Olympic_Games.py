import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
games = pd.read_csv("C:\\Users\\khila\\OneDrive\\Documents\\Business Intelligence II\\Project\\Olympic_Games.csv")

# Group by year and count occurrences
participation = games.groupby('year').size().reset_index(name='count')

# Set year as index
participation.set_index('year', inplace=True)

# ARIMA model
model = ARIMA(participation, order=(5, 1, 0))  # Adjust the order as needed
model_fit = model.fit()

# Future prediction
forecast = model_fit.get_forecast(steps=12)
forecast_df = forecast.conf_int()
forecast_df['forecast'] = model_fit.predict(start=forecast_df.index[0], end=forecast_df.index[-1])

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(participation, label='Actual Participation')
plt.plot(forecast_df['forecast'], label='Forecasted Participation', color='red')
plt.fill_between(forecast_df.index,
                 forecast_df.iloc[:, 0],
                 forecast_df.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast of Olympic Games Participation')
plt.xlabel('Year')
plt.ylabel('Number of Competitions/Entries')
plt.legend()
plt.show()

# Save the forecasted results
forecast_df.to_csv('olympic_participation_forecast.csv', index=True)
