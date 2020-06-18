import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

output = []

for i in range(243):
  rows = 838-243+i
  series = pd.read_csv('project.csv', header=0, nrows=rows, index_col=0, squeeze=True)
  model = ARIMA(series, order=(0,1,1))
  model_fit = model.fit(trend='c', full_output=True, disp=1)
  fore = model_fit.forecast(steps=1)
  output.append(fore[0])

output = np.array(output)

fig = plt.figure()
plt.plot(output,'r')
fig.savefig('hi.png',bbox_inches='tight')


