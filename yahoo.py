import yfinance as yf
import matplotlib.pyplot as plt

albb=yf.Ticker('BABA')
albb_info=albb.info
hist=albb.history(period='max')
hist_close=hist['Close']

plt.figure(figsize=(26,14))
plt.title('albb')
hist_close.plot()
plt.show()
print(albb_info)