import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#%%
def load_data(path):# 对整个数据处理,我们假装不知道"这个时间"以后的数据
    sys.path.insert(0,".")
    data = pd.read_csv(path)

    idx = []
    for i in range(len(data)):
        if data.iloc[i,1]==0 or data.iloc[i,2]==0:
            idx.append(i)

    data = data.drop(idx).reset_index(drop=True)

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

path =  "data/IC_IF/IF/IF2012-2009.csv"
data = load_data(path)

#%%
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

    return intervals

grids = [-9,-6,-3,0,3,6,9]
intervals = grids_to_intervals(grids)

#%%

def find_interval(x,grids): # 判断当前在哪个区间
    grid = len(grids)-1
    if x>=grids[0] and x<=grids[-1]:
        for i in range(len(grids) - 1):
            if x >= grids[i] and x < grids[i + 1]:
                grid = i+1
                break
    elif x<grids[0]:
        grid = 0
    else:
        grid = len(grids)
    return grid


#%%
path_time = "data/phase2/execute_time/time1209.csv"
execute_time = pd.read_csv(path_time)
def do_time_series(data, execute_time,frequency):
    # 首先判断最新时间在哪个区间
    get = -1
    start = -1
    end = -1
    cnt = 0
    for i in range(len(data)-1,-1,-1):
        # 新添加的数据里，必有处在可执行区间里的数才行



        if get==-1:
            for j in range(execute_time.shape[0]):
                if data.loc[i,'30min_idx']>=execute_time.iloc[j,0] and data.loc[i,'30min_idx']<=execute_time.iloc[j,1]:
                    get=j
                    end = i
                    cnt += 1
                    break
        else:
            if data.loc[i,'30min_idx']>=execute_time.iloc[get,0] and data.loc[i,'30min_idx']<=execute_time.iloc[j,1]:
                start = i
                cnt += 1
            else:
                if cnt>=200:
                    return start,end
                else:
                    return -1,-1

    return -1,-1

start,end = do_time_series(data,execute_time,frequency)







#%%
# 按照某个更新频率移动窗口 模拟实时读入数据的过程


def run(data):

    # do something

    return




path = "data/IC_IF/IF/IF2003-1912.csv"
data_org = load_data(path)



frequency = 10   # 频率 可调参数 先设为10min
turns = int(len(data_org)/frequency)

# 初始化
start = 0
end = start+frequency
data_history = data_org[start:end]  # 存储历史数据

for turn in range(turns):

    new_data = data_org[start:end]  # 读入新数据

    # 对一段新数据进行操作的过程
    run(new_data)

    start = start+frequency
    end = start+frequency

    if(turn != 0):
        data_history = data_history.append(new_data)


#%%
def cross_points(data): # 传入更新片段的data
    points = [] # 与网格线相交的点idx，包括真性穿线、假性穿线
    for i in range(1,len(data)):
        if data.loc[i,"grid"] != data.loc[i-1,"grid"]:
            points.append(i)

    return points


def data_process(data, grids, base_line):# 传入不同的grids,传入更新片段的data

    data['gap'] = data['spread'] - data[base_line]
    data['grid'] = data.gap.apply(lambda x: int(find_grid(x, grids)))
    points = cross_points(data)
    return data, points

#def get_new_account(data, points, grids, market, service_rate, max_hold, min_hold, add, ):



'''
def get_account_log_two_points(data, points, grids, market, service_rate, max_hold, min_hold,add):
    data['hold'] = ''
    data['service'] = ''
    data['bag'] = ''
    data['revenue'] = ''
    data['account'] = ''
    service = 0 # 叠加 每点更新
    bag = 0 # 落袋金额
    price_stack = [] # 存储仓内买入、卖空价
    hold = 0 # 总持有数
    cnt = 0 # 网格交叉点计数
    mul = 200 if market == "IC" else 300
    pre_grid = -1 # 上一单所处区间编号
    change = [-1,-1]
    change_hold = 0

    flag = False
    t1 = -1
    t1_grid = -1
    pre_grid_intend = -1
    for i in range(len(data)):
        new_service = 0
        if i == 0:  # 从头开始

            change = [data.loc[i, "grid"], data.loc[i, "grid"]]
            hold = int(len(grids) / 2) - data.loc[i, "grid"]

            # previous price
            pre_gap = grids[data.loc[i, "grid"] - 1] if data.loc[i, "grid"] != 0 else grids[0]
            price = pre_gap + data.loc[i, "spread"] - data.loc[i, "gap"]

            for times in range(abs(hold)):
                price_stack.append(price)

            if data.loc[i, "grid"] > max(change):
                change[0] = max(change)
                change[1] = data.loc[i, "grid"]
            else:
                change[0] = min(change)
                change[1] = data.loc[i, "grid"]

            if i == points[cnt] and cnt != len(points) - 1:
                cnt = cnt + 1
        else:
            if i == points[cnt] and cnt != len(points) - 1:  # 是交叉点 可能可以交易
                # change_hold需要在实际操作中再得出来，否则会有持仓或资金的限制
                pre_grid_intend = nearest(data.loc[i, "grid"], change) # 如果实际交易了，再更新为pre_grid
                intend_change = pre_grid_intend - data.loc[i, "grid"]
                deal = True
                change_hold = 0

                if intend_change > 0:
                    if hold == max_hold:
                        deal = False
                    else:
                        deal = True
                        change_hold = min(intend_change, max_hold - hold)
                elif intend_change < 0:
                    if hold == min_hold:
                        deal = False
                    else:
                        deal = True
                        change_hold = max(intend_change, min_hold - hold)

                # fake change
                change2 = change[:]
                change2[0] = change2[1]
                change2[1] = data.loc[i, "grid"]
                if sorted(change) == sorted(change2) and abs(change2[1] - change2[0]) <= 1:  # 直接冲破两格
                    deal = False

                if change_hold == 0:
                    deal = False

                if change[0] == change[1]:  # 开局
                    if data.loc[i, "grid"] == change[0]:
                        deal = False
                    elif abs(data.loc[i, "grid"] - change[0]) == 1:
                        deal = False
                        change[1] = data.loc[i, "grid"]
                    else:
                        deal = True

                if deal:
                    old_hold = hold
                    hold = hold + change_hold
                    new_service = (data.iloc[i, 1] + data.iloc[i, 2]) * mul * service_rate * abs(change_hold)

                    price = data.loc[i, "spread"]
                    # bail & price append
                    if old_hold * hold > 0:
                        if change_hold * old_hold > 0:  # pay bail
                            for times in range(abs(change_hold)):
                                price_stack.append(price)
                        else:  # no pay
                            for times in range(abs(change_hold)):
                                old_price = price_stack.pop()
                                bag = bag + (price - old_price) * mul if old_hold > 0 else bag + (
                                            old_price - price) * mul
                    elif old_hold * hold == 0:
                        if old_hold == 0:
                            for times in range(abs(change_hold)):
                                price_stack.append(price)
                        else:
                            for times in range(abs(change_hold)):
                                old_price = price_stack.pop()
                                bag = bag + (price - old_price) * mul if old_hold > 0 else bag + (old_price - price) * mul
                    else:
                        for times in range(abs(old_hold)):
                            old_price = price_stack.pop()
                            bag = bag + (price - old_price) * mul if old_hold > 0 else bag + (old_price - price) * mul
                        for times in range(abs(hold)):
                            price_stack.append(price)

                    if data.loc[i, "grid"] > max(change):
                        if abs(change_hold) == 1:
                            change[0] = max(change)
                            change[1] = data.loc[i, "grid"]
                        else:
                            change[0] = data.loc[i, "grid"] - 1
                            change[1] = data.loc[i, "grid"]
                    else:
                        if abs(change_hold) == 1:
                            change[0] = min(change)
                            change[1] = data.loc[i, "grid"]
                        else:
                            change[0] = data.loc[i, "grid"] + 1
                            change[1] = data.loc[i, "grid"]

                cnt = cnt + 1

        # i点account计算
        revenue = data.loc[i,"spread"] * hold - sum(price_stack) if hold>=0 else sum(price_stack)+data.loc[i,"spread"]*hold
        revenue = revenue * mul
        service = service + new_service
        bag = bag - add * mul * abs(change_hold)
        account = revenue + bag - service #- sum(bail_stack)
        data.loc[i,'hold']=hold
        data.loc[i,'service']=new_service
        data.loc[i,'bag']=bag
        data.loc[i,'revenue']=revenue
        data.loc[i,'account']=account
    return data

'''



