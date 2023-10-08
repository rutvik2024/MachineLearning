# -*- coding: utf-8 -*-
"""Rainfall_predict.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FL-IjstB2L2HvmM9YarZ2yVPPwv3Q8zn
"""

# Import required library

# For mathematical operation
import numpy as np

# For data frame manipulation
import pandas as pd

# For data scatter
import matplotlib.pyplot as plt

# Read data from data set
data = pd.read_csv('Rainfall_dataset.csv')

rain_fall = data.iloc[:, 2]
yield_crop = data.iloc[:, 3]
plt.scatter(rain_fall, yield_crop)
plt.show()

rain_fall_mean = np.mean(rain_fall)
yield_crop_mean = np.mean(yield_crop)

numerator = 0
denominator = 0

for i in range(len(rain_fall)):
    numerator += (rain_fall[i] - rain_fall_mean) * (yield_crop[i] - yield_crop_mean)
    denominator = (rain_fall[i] - rain_fall_mean)**2
    
m = numerator/ denominator
c = yield_crop_mean - m*rain_fall_mean

print(m, c)

yield_crop_pred = m*rain_fall + c

plt.scatter(rain_fall, yield_crop)
plt.plot([min(rain_fall), max(rain_fall)], [min(yield_crop_pred), max(yield_crop_pred)], color = 'red')
plt.show()

rain_fall_predic = int(input("Enter Rain_fall in mm"))
predict_yield_crop = m*rain_fall_predic + c

print("Yield crop predicted value is : ",predict_yield_crop)

