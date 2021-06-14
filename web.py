import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima


###############################################
# Do not need this on web
sys.path.insert(0,'.')
path = 'data/web/example.csv'
data_full = pd.read_csv(path)
#%%
data = data_full.iloc[:250,:]
###############################################

#%%
T1 = 180
freq = 10
grids = [-12,-9,-6,-3,0,3,6]

exe_time = pd.DataFrame()
#12-09
exe_time['start'] = [90,180,300]
exe_time['end'] = [100,240,350]
#09-06
exe_time['start'] = [10,250,340]
exe_time['end'] = [80,300,400]

volatility = []
entry_num = 0
min_idx = 1

#%%
# get volatility

def get_volatility(data):
    for i in range(241,len(data)):
        if len(volatility)==180:
            volatility.pop()
        volatility.append(np.std(data.iloc[i-240:i,2]))

get_volatility(data)

#%%
def tag_time(data):
    global entry_num
    global min_idx

    data['30min_idx']=''
    for i in range(len(data)):
        entry_num += 1
        if entry_num%30==0:
            min_idx += 1
        data.loc[i,'30min_idx'] = min_idx


tag_time(data)


#%%
def do_time_series(data, exe_time, T1):
    time_idx = -1 # 执行区间编号
    idx_start = -1 # idx_end往前推一个T1
    idx_end = -1 # 当前数据的结尾
    itv_end = -1

    if len(volatility)<T1: # 波动率的数据不够，时间太靠前，直接pass
        return time_idx,idx_start,idx_end,itv_end

    for i in range(len(exe_time)): # 如果尾部数据在执行区间内，那就在那个区间
        if data.loc[len(data)-1,'30min_idx']>exe_time.iloc[i,0] and data.loc[len(data)-1,'30min_idx']<exe_time.iloc[i,1]:
            time_idx = i
            break

    if time_idx!=-1:
        if data.loc[len(data)-T1,'30min_idx']<exe_time.iloc[time_idx,0]: # 从idx_end倒推的idx_start不在该执行区间
            time_idx = -1
        else:
            idx_end = len(data)
            idx_start = idx_end-T1
            itv_end = data.loc[len(data)-1,'30min_idx']

    return time_idx,idx_start,idx_end,itv_end


time_idx,idx_start,idx_end,itv_end = do_time_series(data,exe_time,T1)

#%%

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

        data2['spread'] = data.loc[self.idx_start:self.idx_end,'spread'].reset_index(drop=True)
        data2['volatility'] = data.loc[self.idx_start:self.idx_end,'index_volatility'].reset_index(drop=True)
        data_train['spread'] = data.loc[self.idx_start+freq:self.idx_end,'spread'].reset_index(drop=True)
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
        dif = (np.mean(self.forecast)-self.data.iloc[-1,0])*shift_zoom
        if dif%0.2>0.1:
            self.shift = dif//0.2*0.2+0.2
        else:
            self.shift = dif//0.2*0.2



        # density
        density_zoom = 10 # 预测波动程度放大倍数
        std = np.std(self.forecast)
        self.density = std # 这里肯定要改




#%%
def simulate(data, exe_time, grids, pre_price, market, service_rate, max_hold, min_hold, add, pre_hold, pre_rvn,
             pre_act, pre_bag, pre_change):  # 模拟滑窗
    turns = len(data) // freq  # 完整历史数据的data
    windows = []

    for turn in range(turns):
        new_data = data.iloc[:freq * (turn + 1), :]  # .reset_index()
        if turn == 0:
            grids, data_p, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = run(new_data, exe_time, grids, pre_price,
                                                                                 market, service_rate, max_hold,
                                                                                 min_hold, add, pre_hold, pre_rvn,
                                                                                 pre_act, pre_bag, True, pre_change)
        else:
            grids, data_p, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = run(new_data, exe_time, grids, pre_price,
                                                                                 market, service_rate, max_hold,
                                                                                 min_hold, add, pre_hold, pre_rvn,
                                                                                 pre_act, pre_bag, False, pre_change)

        windows.append(data_p)

    log = pd.concat(windows)
    return log, grids


def run(new_data, exe_time, grids, pre_price, market, service_rate, max_hold, min_hold, add, pre_hold, pre_rvn, pre_act,
        pre_bag, fresh_start, pre_change):  # t时刻之前的所有数据、包含spread,v,30min_idx

    # t1:ts调整网格（根据new_data)
    # t1~t2:交易

    # new_data √

    # t1:ts调整网格
    # 是否要做ts
    time_idx, idx_start, idx_end, itv_end = do_time_series(new_data, exe_time, T1)
    fc_len = 0 if time_idx == -1 else freq
    # object gridSummary描述根据t1~t2更新网格的情况
    one = gridSummary(new_data, fc_len=fc_len, time_idx=time_idx, idx_start=idx_start, idx_end=idx_end)
    grids = [x + one.shift for x in grids]

    # t1~t2交易
    data_trade = new_data[-freq:].reset_index()
    data, points = data_process(data_trade, grids, 'avg3')
    data = data.drop(columns=["index"])

    # data_p = data_p.iloc[:,[1,2,3,4,5,15,17,18]]

    if fresh_start:
        data, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = get_account_log(data, points, grids, pre_price, market,
                                                                                service_rate, max_hold, min_hold, add,
                                                                                True, pre_hold, pre_rvn, pre_act,
                                                                                pre_bag, pre_change)
    else:
        data, pre_hold, pre_rvn, pre_act, pre_bag, pre_change = get_account_log(data, points, grids, pre_price, market,
                                                                                service_rate, max_hold, min_hold, add,
                                                                                False, pre_hold, pre_rvn, pre_act,
                                                                                pre_bag, pre_change)

    return grids, data, pre_hold, pre_rvn, pre_act, pre_bag, pre_change






