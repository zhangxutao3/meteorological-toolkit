import pandas as pd
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from interpolate.point2grid import IDW
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\DELL\Desktop\2018010100.csv")
df = df.dropna()
lon_obs = df["Lons"].values
lat_obs = df["Lats"].values
value = df["PM2.5"].values

lon = np.arange(70, 140, 0.5)
lat = np.arange(10, 50, 0.5)

data = IDW(lon_obs, lat_obs, value, lon, lat, device="cuda").interpolate()
print(data.shape)

plt.imshow(data)
plt.show()
