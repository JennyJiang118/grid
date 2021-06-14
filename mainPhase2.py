import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pmdarima.arima import auto_arima

#%%
# 参数设置
T1 = 180
freq = 10
grids = [-9,-6,-3,0,3,6,9]
base_line = 'avg3'
service_rate = 0.000026  # 手续费
max_hold = 4 # 仓位上限
min_hold = -4 # 仓位下限
add = 0.4
market = "IF"

pre_price = []
pre_act = 0
pre_rvn = 0
pre_itv = -1
pre_hold = 0
pre_bag = 0

path =  "data/IC_IF/IF/IF2012-2009.csv"
v_path = 'data/index_volatility/IF2012-2009.csv'
exe_path = "data/phase2/execute_time/time1209.csv"


#%%
# 读取数据

def load_data(path):# 对整个数据处理,我们假装不知道"这个时间"以后的数据
    sys.path.insert(0,".")
    data = pd.read_csv(path,index_col=False)

    idx = []
    for i in range(len(data)):
        if data.iloc[i,1]==0 or data.iloc[i,2]==0:
            idx.append(i)

    data = data.drop(idx)#.reset_index(drop=True)

    data['30min_idx'] = ''
    min_idx = 1
    for i in range(0,len(data)):
        if i%30==0:
            min_idx += 1
        data.loc[i,'30min_idx'] = min_idx

    data["avg3"] = ''
    data["avg5"] = ''
    for i in range(720,len(data)):
        data.loc[i,"avg3"] = round(np.mean(data.loc[i-720:i,"spread"])/0.2)*0.2
    for i in range(1200,len(data)):
        data.loc[i,"avg5"] = round(np.mean(data.loc[i-1200:i,"spread"])/0.2)*0.2

    idx.clear()
    for i in range(len(data)):
        if data.loc[i,"avg3"]=='' or data.loc[i,"avg5"]=='':
            idx.append(i)
    data = data.drop(idx).reset_index(drop=True)
    return data

def load_volatility(path): # 读入波动率数据
    volatility_data = pd.read_csv(v_path)
    return volatility_data

def load_exe(path): # 读入可执行区间数据
    execute_time = pd.read_csv(path)
    return execute_time

def load(data, volatility_data):
    data = pd.merge(data, volatility_data, on='datetime', how='inner')
    return data


data = load_data(path)
volatility_data = load_volatility(v_path)
exe_time = load_exe(exe_path)
data = load(data,volatility_data) # 这里的data是完整的历史data，模拟真实情况时需要滑窗读入

#%%
# 时间序列代码

def do_time_series(data, exe_time,T1):
    # 首先判断最新时间在哪个区间
    # 可以做ts的必要条件：data尾部数据在可执行区间内，记尾部可执行数据标号为idx_end
    # 第二步，判断idx_start = idx_end-T1 是否在"该"区间内
    # idx_end是当前数据的结尾
    # idx_start是idx_end往前推一个T1（最少训练长度）
    time_idx = -1 #执行区间编号
    for i in range(len(exe_time)):#如果尾部数据在执行区间内，在哪个区间
        if data.loc[len(data)-1,'30min_idx']>exe_time.iloc[i,0] and data.loc[len(data)-1,'30min_idx']<exe_time.iloc[i,1]:
            time_idx = i
            break

    if time_idx!=-1:
        if len(data)<T1: #时间长度不够
            time_idx = -1
        else:
            if data.loc[len(data)-T1,'30min_idx']<exe_time.iloc[time_idx,0]: #从idx_end倒退的idx_start不在该执行区间
                time_idx = -1
            else:
                idx_start = len(data)-T1
                idx_end = len(data)
                itv_end = data.loc[len(data)-1,'30min_idx']


    if time_idx == -1:
        idx_start,idx_end,itv_end = -1,-1,-1

    return time_idx,idx_start,idx_end,itv_end




# 调整完了之后，执行区间结束，需要调回去吗？
# 不用，spread已经变过了
# 要变
# 两个都试一下，比较

class gridSummary():
    # 当前时间，根据时间序列（如果有的话，没有就是0）应该作出的网格调整
    def __init__(self,data,fc_len=0,time_idx=0,idx_start=0,idx_end=0):
        self.data = data
        self.train = None
        self.fc_len = fc_len
        self.t_len = 180
        self.time_idx = time_idx
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.forecast = None
        self.shift = 0 #跟之前的网格相比，应该偏移的量
        self.density = 3 #网格间隙
        if self.fc_len!=0:
            self.split()
            self.arima()
            self.change()

    def split(self):#data只保留spread,index_volatility
        data2 = pd.DataFrame()
        data_train = pd.DataFrame()

        data2['spread'] = data.loc[self.idx_start:self.idx_end,'spread_x'].reset_index(drop=True)
        data2['volatility'] = data.loc[self.idx_start:self.idx_end,'index_volatility'].reset_index(drop=True)
        data_train['spread'] = data.loc[self.idx_start+freq:self.idx_end,'spread_x'].reset_index(drop=True)
        data_train['volatility'] = data.loc[self.idx_start:self.idx_end-freq,'index_volatility'].reset_index(drop=True)

        self.data = data2
        self.train = data_train



    def arima(self):
        model = auto_arima(y=self.train['spread'], x=self.train['volatility'])
        self.forecast = model.predict(x=self.data.loc[-freq:,'volatility'],n_periods = freq)


    def change(self):
        '''
        根据forecast结果，调整shift & density

        shift: forecast与one.data.iloc[-1,0]比较大小
        density: forecast预测出来的波动情况
        :return:
        shift, density改变
        '''

        # shift
        shift_zoom = 0.1 # 预测差异放大倍数
        dif = np.mean(self.forecast)-self.data.iloc[-1,0]*shift_zoom
        if dif%0.2>0.1:
            self.shift = self.data.iloc[-1,0]-(self.data.iloc[-1,0]%0.2)+1
        else:
            self.shift = self.data.iloc[-1,0]-(self.data.iloc[-1,0]%0.2)

        # density
        density_zoom = 10 # 预测波动程度放大倍数
        std = np.std(self.forecast)
        self.density = std # 这里肯定要改

# array([-19.66086893, -19.66086893, -19.66086893, -19.66086893, -19.66086893])
# 真实（未来的，现在未知）[-19,-20.8,-22,-20.6,-23]
# one.data.iloc[-1,0]：-19.8
# 嘶 预测的有点问题，预测价差未来会-19.8-->-19.6即上涨，但实际下降了
# --->
# 试一下：
# 1. 增大freq:  因为预测的时间太短，数据量不够，加上波动率的滞后不是很准确，可能直接把这5分钟冲走了
# 2. 改变T1

# freq 可能要设置的大一些，5min就五个数据，
# 一是很难预测出什么波动
# 二是就算有波动，根据two-points也至少有连续两个点超过才能交易，5个点又能容纳下多少交易呢


#%%
# 交易过程t1~t2
'''
记录网格设置：grids
记录现在位置：intervals
'''
def grids_to_intervals(grids):
    M = 10000  # 表示无限

    n = len(grids)
    intervals = []

    # 初始化
    for i in range(n + 1):
        m = [0, 0]
        intervals.append(m)

    for i in range(n - 1):
        floor = grids[i]
        ceiling = grids[i + 1]
        intervals[i + 1][0] = floor
        intervals[i + 1][1] = ceiling

    intervals[0][0] = M * (-1)
    intervals[0][1] = grids[0]

    intervals[-1][0] = grids[-1]
    intervals[-1][1] = M

    return pd.DataFrame(intervals)



def find_interval(x,grids): # 判断当前在哪个区间
    itv = len(grids)-1
    if x>=grids[0] and x<=grids[-1]:
        for i in range(len(grids) - 1):
            if x >= grids[i] and x < grids[i + 1]:
                itv = i+1
                break
    elif x<grids[0]:
        itv = 0
    else:
        itv = len(grids)
    return itv # itv标号



def cross_points(data): # 传入更新片段的data
    points = [] # 与网格线相交的点idx，包括真性穿线、假性穿线
    for i in range(1,len(data)): # 这里！万一section2的头和section1的尾有itv不同？
        if data.loc[i,"itv"] != data.loc[i-1,"itv"]:
            points.append(i)

    return points


def data_process(data, grids, base_line):# 传入不同的grids,传入更新片段的data

    data['gap'] = data['spread_x'] - data[base_line]
    data['itv'] = data.gap.apply(lambda x: int(find_interval(x, grids)))
    points = cross_points(data)
    return data, points


def nearest(x,change):
    # 求之前是在哪条线(itv）买入的
    choose = change[0] if abs(change[0]-x)<abs(change[1]-x) else change[1]
    return choose



def get_account_log(data, points, grids, pre_price, market, service_rate, max_hold, min_hold,add,fresh_start,pre_hold,pre_rvn,pre_act,pre_bag,pre_change):
    data['hold'] = ''
    data['service'] = ''
    data['bag'] = ''
    data['revenue'] = ''
    data['account'] = ''
    service = 0 # 叠加 每点更新
    bag = 0 # 落袋金额
    #hold = -99 # 总持有数
    cnt = 0 # 网格交叉点计数
    mul = 200 if market == "IC" else 300
    change = pre_change
    change_hold = 0


    pre_grid_intend = -1
    for i in range(len(data)):
        new_service = 0
        if i==0 and fresh_start:  # 整个data的最开始

            # 初始化持仓
            change = [data.loc[i, "itv"], data.loc[i, "itv"]]
            hold = int(len(grids) / 2) - data.loc[i, "itv"]

            # previous price
            pre_gap = grids[data.loc[i, "itv"] - 1] if data.loc[i, "itv"] != 0 else grids[0]
            price = pre_gap + data.loc[i, "spread_x"] - data.loc[i, "gap"]

            # hold=-3
            # change=[6,6]


            for times in range(abs(hold)):
                pre_price.append(price)
            if data.loc[i, "itv"] > max(change):
                change[0] = max(change)
                change[1] = data.loc[i, "itv"]
            else:
                change[0] = min(change)
                change[1] = data.loc[i, "itv"]

            if len(points)>0:
                if i == points[cnt] and cnt != len(points) - 1:
                    cnt = cnt + 1

            pre_hold = hold
            print('change01',change)


        else:
            if len(points)>0:
                if i == points[cnt] and cnt != len(points) - 1:  # 是交叉点 可能可以交易
                    # change_hold需要在实际操作中再得出来，否则会有持仓或资金的限制

                    # change=[5,4]
                    # 4 -> 5
                    print('change',change)
                    pre_grid_intend = nearest(data.loc[i, "itv"], change) # 如果实际交易了，再更新为pre_grid
                    print('pre grid intend',pre_grid_intend)

                    intend_change = pre_grid_intend - data.loc[i, "itv"]


                    deal = True
                    change_hold = 0

                    if intend_change > 0:
                        if pre_hold == max_hold:
                            deal = False
                        else:
                            deal = True
                            change_hold = min(intend_change, max_hold - pre_hold)
                    elif intend_change < 0:
                        if pre_hold == min_hold:
                            deal = False
                        else:
                            deal = True
                            change_hold = max(intend_change, min_hold - pre_hold)

                    # fake change
                    change2 = change[:]
                    change2[0] = change2[1]
                    change2[1] = data.loc[i, "itv"]
                    if sorted(change) == sorted(change2) and abs(change2[1] - change2[0]) <= 1:  # 直接冲破两格
                        deal = False

                    if change_hold == 0:
                        deal = False

                    if change[0] == change[1]:  # 开局
                        if data.loc[i, "itv"] == change[0]:
                            deal = False
                        elif abs(data.loc[i, "itv"] - change[0]) == 1:
                            deal = False
                            change[1] = data.loc[i, "itv"]
                        else:
                            deal = True

                    if deal:
                        old_hold = pre_hold
                        pre_hold = pre_hold + change_hold
                        new_service = (data.iloc[i,1] + data.iloc[i, 2]) * mul * service_rate * abs(change_hold)
                        price = data.loc[i, "spread_x"]

                        # bail & price append
                        if old_hold * pre_hold > 0:
                            if change_hold * old_hold > 0:  # pay bail
                                for times in range(abs(change_hold)):
                                    pre_price.append(price)
                            else:  # no pay
                                for times in range(abs(change_hold)):
                                    old_price = pre_price.pop()
                                    pre_bag = pre_bag + (price - old_price) * mul if old_hold > 0 else pre_bag + (
                                            old_price - price) * mul
                        elif old_hold * pre_hold == 0:
                            if old_hold == 0:
                                for times in range(abs(change_hold)):
                                    pre_price.append(price)
                            else:
                                for times in range(abs(change_hold)):
                                    old_price = pre_price.pop()
                                    pre_bag = pre_bag + (price - old_price) * mul if old_hold > 0 else pre_bag + (old_price - price) * mul
                        else:
                            for times in range(abs(old_hold)):
                                old_price = pre_price.pop()
                                pre_bag = pre_bag + (price - old_price) * mul if old_hold > 0 else pre_bag + (old_price - price) * mul
                            for times in range(abs(pre_hold)):
                                pre_price.append(price)

                        if data.loc[i, "itv"] > max(change):
                            if abs(change_hold) == 1:
                                change[0] = max(change)
                                change[1] = data.loc[i, "itv"]
                            else:
                                change[0] = data.loc[i, "itv"] - 1
                                change[1] = data.loc[i, "itv"]
                        else:
                            if abs(change_hold) == 1:
                                change[0] = min(change)
                                change[1] = data.loc[i, "itv"]
                            else:
                                change[0] = data.loc[i, "itv"] + 1
                                change[1] = data.loc[i, "itv"]

                    cnt = cnt + 1

            else:
                change_hold=0



        revenue = data.loc[i,"spread_x"] * pre_hold - sum(pre_price) if pre_hold>=0 else sum(pre_price)+data.loc[i,"spread_x"]*pre_hold
        revenue = revenue * mul
        service = service + new_service
        pre_bag = pre_bag - add * mul * abs(change_hold)
        account = revenue + pre_bag - service
        data.loc[i,'hold']=pre_hold
        data.loc[i,'service']=new_service
        data.loc[i,'bag']=pre_bag
        data.loc[i,'revenue']=revenue
        data.loc[i,'account']=account

        pre_rvn = revenue
        pre_act = account
        pre_change = change




    return data,pre_hold,pre_rvn,pre_act,pre_bag,pre_change


# 调试时，第一次运行记得pre_price等参数重新初始化


#%%
# 滑窗读入

def simulate(data,exe_time,grids,pre_price,market,service_rate,max_hold,min_hold,add,pre_hold,pre_rvn,pre_act,pre_bag,pre_change): # 模拟滑窗
    turns = len(data)//freq # 完整历史数据的data
    windows = []


    for turn in range(turns):
        new_data= data.iloc[:freq*(turn+1),:]#.reset_index()
        if turn==0:
            grids, data_p, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = run(new_data, exe_time, grids, pre_price, market,service_rate, max_hold, min_hold,add, pre_hold,pre_rvn,pre_act,pre_bag,True,pre_change)
        else:
            grids, data_p, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = run(new_data, exe_time, grids, pre_price, market,service_rate, max_hold, min_hold,add, pre_hold,pre_rvn,pre_act,pre_bag,False,pre_change)

        windows.append(data_p)

    log = pd.concat(windows)
    return log,grids


def run(data, exe_time, grids, pre_price, market,service_rate, max_hold, min_hold,add, pre_hold,pre_rvn,pre_act,pre_bag,fresh_start,pre_change): # t时刻之前的所有数据、包含spread,v,30min_idx

    # t1:ts调整网格（根据new_data)
    # t1~t2:交易



    # t1:ts调整网格
    # 是否要做ts
    time_idx, idx_start, idx_end, itv_end = do_time_series(data, exe_time, T1)
    fc_len = 0 if time_idx == -1 else freq
    # object gridSummary描述根据t1~t2更新网格的情况
    one = gridSummary(data, fc_len=fc_len, time_idx=time_idx, idx_start=idx_start, idx_end=idx_end)
    grids = [x+one.shift for x in grids]

    # t1~t2交易
    data_trade = data[-freq:].reset_index()
    data_p,points = data_process(data_trade,grids,'avg3')

    data_p = data_p.iloc[:,[1,2,3,4,5,15,17,18]]

    if fresh_start:
        data_p, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = get_account_log(data_p, points, grids, pre_price, market, service_rate, max_hold, min_hold,add,True,pre_hold,pre_rvn,pre_act,pre_bag,pre_change)
    else:
        data_p, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = get_account_log(data_p, points, grids, pre_price, market, service_rate, max_hold, min_hold,add,False,pre_hold,pre_rvn,pre_act,pre_bag,pre_change)


    return grids, data_p, pre_hold, pre_rvn, pre_act, pre_bag, pre_change


#data_tmp = data[:1000]

log,new_grids = simulate(data,exe_time,grids,pre_price,market,service_rate,max_hold,min_hold,add,pre_hold,pre_rvn,pre_act,pre_bag,[-1,-1])


#%%
def plot(account_list):
    x = np.linspace(1,len(log),len(log))
    plt.plot(x, account_list)
    plt.show()

plot(log['account'])


