# 指数波动率
library(urca)
spread = IF2003.1912$spread
volatility = IF2003.1912$index_volatility
spread = na.omit(spread)
volatility = na.omit(volatility)

plot.ts(spread)
plot.ts(volatility)

library(tseries)
adf.test(volatility)
adf.test(spread)

spread.dif1 = diff(spread,differences = 1)
plot(spread.dif1)
adf.test(spread.dif1)

# 白噪声检验
for(i in 1:3) print(Box.test(spread,type="Ljung-Box",lag=6*i))
for(i in 1:3) print(Box.test(volatility,type="Ljung-Box",lag=6*i))
#非白噪声


# 自相关图、偏自相关图
# spread:
# ARIMA(9,1,1)
acf(spread,lwd = 2, col=4)
acf(spread.dif1, lwd=2, col=4)
pacf(spread.dif1,lwd=2,col=4)

# volatility:
# ARIMA(9,1,2)
volatility_dif1 = diff(volatility,differences = 1)
volatility_dif2 = diff(volatility,differences = 2)
acf(volatility_dif2,lwd=2,col=4)
pacf(volatility_dif2,lwd=2,col=4)


# 对模型进行定阶
spread_fit = arima(spread,order=c(11,1,1))

# 残差白噪声检验
for(i in 1:3) print(Box.test(spread_fit$residuals,lag=6*i))
# 通过

# 参数显著性检验
t1 = -0.8580/0.245
pt(t1,df=36,lower.tail = F)




