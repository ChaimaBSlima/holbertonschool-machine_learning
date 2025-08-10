#!/usr/bin/env python3
"""Script to visualize a transformed pd.DataFrame"""


import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__("2-from_file").from_file

df = from_file("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", ",")

# Remove the Weighted_Price column
df = df.drop(columns=["Weighted_Price"])

# Rename the column Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert the timestamp values to date values
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Index the data frame on Date
df = df.set_index("Date")

# Missing values in Close should be set to the
# previous row value (forward fill)
df["Close"] = df["Close"].ffill()

# Missing values in High, Low, Open should be set to the same row's Close value
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])

# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

# Filter data from 2017 and beyond
df_2017_beyond = df[df.index >= "2017"]

# Resample to daily intervals and group the values of the same day
df_daily = df_2017_beyond.resample("D").agg(
    {
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
    }
)

# Plot all columns
plt.plot(df_daily.index, df_daily["High"], label="High")
plt.plot(df_daily.index, df_daily["Low"], label="Low")
plt.plot(df_daily.index, df_daily["Open"], label="Open")
plt.plot(df_daily.index, df_daily["Close"], label="Close")
plt.plot(df_daily.index, df_daily["Volume_(BTC)"], label="Volume_(BTC)")
plt.plot(df_daily.index, df_daily["Volume_(Currency)"],
         label="Volume_(Currency)")

plt.xlabel("Date")
plt.legend()

# Format x-axis to show months and years like "Jan 2017", "Apr", etc.
ax = plt.gca()
# Get every 3rd month for major ticks
months = pd.date_range(start="2017-01", end="2019-02", freq="3MS")
ax.set_xticks(months)
# Format the labels
labels = []
for i, date in enumerate(months):
    if i == 0 or date.year != months[i - 1].year:
        labels.append(f"{date.strftime('%b')}\n{date.year}")
    else:
        labels.append(date.strftime("%b"))
ax.set_xticklabels(labels)

plt.show()

# Print the transformed DataFrame
print(df_daily)

