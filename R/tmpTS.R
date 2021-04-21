library(fpp2)
library(forecast)
library(ggplot2)
library(tseries)

#（2000，3700）
#（5000，5500）
t0=5000
t1=t0+240*3
t2=t1+240
#t2=nrow(data)

data_ts = ts(data$spread)
plot(data_ts)

train = data[t0:t1,]
test = data[t1+1:t2,]


model = auto.arima(train$spread,xreg = train$index_volatility)
summary(model)
checkresiduals(model)

# 根据波动率数据预测【同一时间的】价差
# 后续波动率提前

# 可行的forecast
forecast = predict(model,newxreg = test$index_volatility)

#spread_compare = data.frame(cbind(forecast$pred,test$spread))

# 官网做法
# upper prediction intervals are not finite
fcast = forecast(model,xreg = test$index_volatility)
autoplot(fcast)

#real_ts = ts(data[t0:t2,]$spread)
#plot(real_ts)
spread_compare = data.frame(cbind(fcast$mean,test$spread))
forecast_value = ts(fcast$mean)


# forecast画图
# model %>% forecast(xreg = test$index_volatility) %>% autoplot()


write.csv(spread_compare,file='spreadcompare.csv')


