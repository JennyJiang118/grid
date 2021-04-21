
#对一个合约，得到p,q的时间序列
#对所有合约重复操作，得到所有的p,q时间序列
#找出所有合约共性的平稳pq时间段，在该时间段内使用ARIMA预测趋势

#T1=200
#step=10



library(fpp2)
library(forecast)
library(ggplot2)
library(tseries)

step=30
pc=c()
dc=c()
qc=c()
T = 240
for (i in seq(0,nrow(data)-T,step)){
  start=i
  end=i+T
  data.reg = data[start:end,]
  model = auto.arima(data.reg$spread,xreg = data.reg$index_volatility)
  # test阶段用来回归的volatility怎么办
  # 哦原来不需要，我只要得到train即data.reg阶段的pq就行了
  pdq = as.numeric(arimaorder(model))
  pc = c(pc,pdq[1])
  dc = c(dc,pdq[2])
  qc = c(qc,pdq[3])
}

pdq = data.frame(cbind(pc,dc,qc))

pts = ts(pc)
dts = ts(dc)
qts = ts(qc)
plot(pts)
plot(dts)
plot(qts)



write.csv(pdq,file='pdq1906-1903.csv')


