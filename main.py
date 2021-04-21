import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def load_data(path):
    sys.path.insert(0,".")
    data = pd.read_csv(path)

    idx = []
    for i in range(len(data)):
        if data.iloc[i,1]==0 or data.iloc[i,2]==0:
            idx.append(i)

    data = data.drop(idx).reset_index(drop=True)

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
    #data = data.dropna(axis=0, how='any').reset_index()
    return data


#%%
def data_process(data, grids, base_line):
    data['gap'] = data['spread'] - data[base_line]
    data['grid'] = data.gap.apply(lambda x: int(find_grid(x, grids)))
    #data['grid_sell'] = data.gap.apply(lambda x: int(find_grid(x, grids_sell)))
    #data['grid_buy'] = data.gap.apply(lambda x: int(find_grid(x, grids_buy)))

    #points, points_sell, points_buy = cross_points(data)
    points = cross_points(data)
    return data, points#, points_sell, points_buy


def find_grid(x,grids):
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


def cross_points(data):
    points = [] # 与网格线相交的点idx
    #points_sell = []
    #points_buy = []
    for i in range(1,len(data)):
        if data.loc[i,"grid"] != data.loc[i-1,"grid"]:
            points.append(i)
        #if data.loc[i,'grid_sell'] != data.loc[i-1,'grid_sell']:
         #   points_sell.append(i)
        #if data.loc[i,'grid_buy'] != data.loc[i-1,'grid_buy']:
         #   points_buy.append(i)
    return points#, points_sell, points_buy



#---------------------------get revenue--------------------------#
#%%

def nearest(x,change):
    # 求之前是在哪条线买入的
    choose = change[0] if abs(change[0]-x)<abs(change[1]-x) else change[1]
    return choose


def get_account(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold, add):
    service = 0 # 叠加 每点更新
    bag = 0 # 落袋金额
    price_stack = [] # 存储仓内买入、卖空价
    hold = 0 # 总持有数
    cnt = 0 # 网格交叉点计数
    mul = 200 if market == "IC" else 300
    pre_grid = 1 # 上一单所处网格
    change = [-1,-1]
    account_list = []
    change_hold = 0
    for i in range(len(data)):
        new_service = 0
        if i == 0:  # 从头开始

            change =[data.loc[i,"grid"],data.loc[i,"grid"]]
            hold = int(len(grids) / 2) - data.loc[i,"grid"]

            # previous price
            pre_gap = grids[data.loc[i,"grid"]-1] if data.loc[i,"grid"]!=0 else grids[0]
            price = pre_gap+data.loc[i,"spread"]-data.loc[i,"gap"]

            for times in range(abs(hold)):
                price_stack.append(price)


            if data.loc[i, "grid"] > max(change):
                change[0] = max(change)
                change[1] = data.loc[i, "grid"]
            else:
                change[0] = min(change)
                change[1] = data.loc[i, "grid"]

            if i == points[cnt] and cnt != len(points)-1:
                cnt = cnt + 1
        else:
            if i == points[cnt] and cnt != len(points)-1: # 是交叉点 可能可以交易
                # change_hold需要在实际操作中再得出来，否则会有持仓或资金的限制
                pre_grid = nearest(data.loc[i,"grid"],change)
                intend_change = pre_grid-data.loc[i,"grid"]
                deal = True
                change_hold = 0

                if intend_change>0:
                    if hold==max_hold: deal = False
                    else:
                        deal = True
                        change_hold = min(intend_change, max_hold-hold)
                elif intend_change<0:
                    if hold==min_hold: deal = False
                    else:
                        deal = True
                        change_hold = max(intend_change, min_hold-hold)

                # fake change
                change2 = change[:]
                change2[0] = change2[1]
                change2[1] = data.loc[i,"grid"]
                if sorted(change)==sorted(change2) and abs(change2[1]-change2[0])<=1: # 直接冲破两格
                    deal = False

                if change_hold==0:
                    deal = False

                if change[0]==change[1]: # 开局
                    if data.loc[i,"grid"]==change[0]:
                        deal=False
                    elif abs(data.loc[i,"grid"]-change[0])==1:
                        deal=False
                        change[1]=data.loc[i,"grid"]
                    else:
                        deal = True

                if deal:
                    old_hold = hold
                    hold = hold + change_hold
                    new_service = (data.iloc[i, 1] + data.iloc[i, 2]) * mul * service_rate * abs(change_hold)

                    price = data.loc[i,"spread"]
                    # bail & price append
                    if old_hold * hold > 0:
                        if change_hold * old_hold > 0:  # pay bail
                            for times in range(abs(change_hold)):
                                price_stack.append(price)
                        else:  # no pay
                            for times in range(abs(change_hold)):
                                old_price = price_stack.pop()
                                bag = bag + (price - old_price)*mul if old_hold>0 else bag + (old_price-price)*mul
                    elif old_hold * hold == 0:
                        if old_hold == 0:
                            for times in range(abs(change_hold)):
                                price_stack.append(price)
                        else:
                            for times in range(abs(change_hold)):
                                old_price = price_stack.pop()
                                bag = bag + (price - old_price)*mul if old_hold>0 else bag + (old_price-price)*mul
                    else:
                        for times in range(abs(old_hold)):
                            old_price = price_stack.pop()
                            bag = bag + (price - old_price) * mul if old_hold > 0 else bag + (old_price - price) * mul
                        for times in range(abs(hold)):
                            price_stack.append(price)

                    if data.loc[i,"grid"]>max(change):
                        if abs(change_hold)==1:
                            change[0]=max(change)
                            change[1]=data.loc[i,"grid"]
                        else:
                            change[0]=data.loc[i,"grid"]-1
                            change[1]=data.loc[i,"grid"]
                    else:
                        if abs(change_hold)==1:
                            change[0]=min(change)
                            change[1]=data.loc[i,"grid"]
                        else:
                            change[0]=data.loc[i,"grid"]+1
                            change[1]=data.loc[i,"grid"]

                cnt = cnt + 1



        # i点account计算
        revenue = data.loc[i,"spread"] * hold - sum(price_stack) if hold>=0 else sum(price_stack)+data.loc[i,"spread"]*hold
        revenue = revenue * mul
        service = service + new_service
        bag = bag - add * mul * abs(change_hold)
        account = revenue + bag - service #- sum(bail_stack)
        account_list.append(account)
    return account_list

def get_account_log_sellbuy(data, points_sell, points_buy, market, grids, grids_sell, grids_buy, service_rate, max_hold, min_hold):
    data['hold'] = ''
    data['service'] = ''
    data['bag'] = ''
    data['revenue'] = ''
    data['account'] = ''
    service = 0  # 叠加 每点更新
    bag = 0  # 落袋金额
    price_stack = []  # 存储仓内买入、卖空价
    hold = 0  # 总持有数
    cnt_sell = 0  # sell网格交叉点计数总数
    cnt_buy = 0
    mul = 200 if market == "IC" else 300
    pre_grid = 1  # 上一单所处网格
    change_sell = [-1, -1]
    change_buy = [-1, -1]
    change_hold = 0
    # 买卖分离，唯一的联系是hold, price_stack


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

            tmp = max(change_sell)
            change_sell[0] = tmp
            change_sell[1] = data.loc[i, 'grid_sell']
            change_sell[0] = tmp
            change_sell[1] = data.loc[i, 'grid_buy']

            if i == points_buy[cnt_buy]:
                cnt_buy = cnt_buy + 1
            if i == points_sell[cnt_sell]:
                cnt_sell = cnt_sell + 1

        else:
            # points_sell交叉点 可能可以交易
            if i == points_sell[cnt_sell] and cnt_sell != len(points_sell) - 1:
                # change_hold需要在实际操作中再得出来，否则会有持仓或资金的限制
                pre_grid = nearest(data.loc[i, "grid_sell"], change_sell)
                intend_change = pre_grid - data.loc[i, "grid_sell"]
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
                change2 = change_sell[:]
                change2[0] = change2[1]
                change2[1] = data.loc[i, "grid_sell"]
                if sorted(change_sell) == sorted(change2) and abs(change2[1] - change2[0]) <= 1:  # 直接冲破两格
                    deal = False

                if change_hold == 0:
                    deal = False

                if change_sell[0] == change_sell[1]:  # 开局
                    if data.loc[i, "grid_sell"] == change_sell[0]:
                        deal = False
                    elif abs(data.loc[i, "grid_sell"] - change_sell[0]) == 1:
                        deal = False
                        change_sell[1] = data.loc[i, "grid_sell"]
                    else:
                        deal = True

                if deal:
                    old_hold = hold
                    hold = hold + change_hold
                    new_service = (data.iloc[i, 1] + data.iloc[i, 2]) * mul * service_rate * abs(change_hold)

                    price = data.loc[i, "spread"]
                    # bail & price append
                    # 本来就是sell!
                    # sell: change_hold<0
                    if old_hold * hold > 0: # 卖空加仓
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
                                bag = bag + (price - old_price) * mul if old_hold > 0 else bag + (
                                        old_price - price) * mul
                    else:
                        for times in range(abs(old_hold)):
                            old_price = price_stack.pop()
                            bag = bag + (price - old_price) * mul if old_hold > 0 else bag + (old_price - price) * mul
                        for times in range(abs(hold)):
                            price_stack.append(price)

                    if data.loc[i, "grid_sell"] > max(change_sell):
                        if abs(change_hold) == 1:
                            change_sell[0] = max(change_sell)
                            change_sell[1] = data.loc[i, "grid_sell"]
                            if data.loc[i,'grid_buy'] > max(change_buy):
                                change_buy[0] = max(change_buy)
                                change_buy[1] = data.loc[i,'grid_buy']

                        else:
                            change_sell[0] = data.loc[i, "grid_sell"] - 1
                            change_sell[1] = data.loc[i, "grid_sell"]

                    else:
                        if abs(change_hold) == 1:
                            change[0] = min(change)
                            change[1] = data.loc[i, "grid"]
                        else:
                            change[0] = data.loc[i, "grid"] + 1
                            change[1] = data.loc[i, "grid"]

                cnt_sell = cnt_sell + 1

        # i点account计算
        revenue = data.loc[i, "spread"] * hold - sum(price_stack) if hold >= 0 else sum(price_stack) + data.loc[
            i, "spread"] * hold
        revenue = revenue * mul
        service = service + new_service
        account = revenue + bag - service  # - sum(bail_stack)
        data.loc[i, 'hold'] = hold
        data.loc[i, 'service'] = new_service
        data.loc[i, 'bag'] = bag
        data.loc[i, 'revenue'] = revenue
        data.loc[i, 'account'] = account
    return data

def get_account_log(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold,add):
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
    pre_grid = 1 # 上一单所处网格
    change = [-1,-1]
    change_hold = 0


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
                pre_grid = nearest(data.loc[i, "grid"], change)
                intend_change = pre_grid - data.loc[i, "grid"]
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


def get_account_log_two_points(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold,add):
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
    pre_grid = -1 # 上一单所处网格
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
def nearest_adjust(grids,grid_max, grid_min):
    twos = np.linspace(int(grid_min),int(grid_max),(int(grid_max)-int(grid_min))*5+1)
    idx = 0
    for i in range(len(grids)):
        for j in range(idx,len(twos)):
            if grids[i]==twos[j]:
                idx = j
                break
            elif grids[i]>twos[j]:
                continue
            else:
                idx = j-1
                grids[i]=twos[j-1]
                break
    return grids

def plot2(len, account_list,grid_max, grid_min, grid_num):
    x = np.linspace(1, len, len)
    #x = data['datetime']
    plt.title("grid max="+str(grid_max)+"  grid min="+str(grid_min)+"  grid num="+str(grid_num))
    plt.plot(x, account_list)
    plt.show()

def plot(len, account_list,grids):
    x = np.linspace(1, len, len)
    #x = data['datetime']
    plt.title("grids:"+str(grids))
    plt.plot(x, account_list)
    plt.show()

def plot_bp(len, revenue_list, lr, EPOCH,grids):
    x = np.linspace(1,len,len)
    plt.title("bp revenue "+"lr="+str(lr)+" EPOCH="+str(EPOCH)+" grids="+str(grids))
    plt.plot(x, revenue_list)
    plt.show()

def plot_run(len, revenue_list, grids):
    x = np.linspace(1, len, len)
    plt.title("grids:"+str(grids))
    plt.plot(x, revenue_list)
    plt.show()

#%%

def train_params(path, market, service_rate, bail_rate, max_hold, min_hold, base_line,grids,add):
    # ------------------------------human set grids----------------------#
    '''
    这里处理的是人为给定的grids
    使用自动调参时可以不运行
    '''


    data_org = load_data(path)
    data, points = data_process(data_org, grids, base_line)
    data = get_account_log(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold, add)
    account_list = data['account']
    plot(len(data), account_list, grids) #!!!!!!!!!

    # describe data
    gap_max = max(data['gap'])
    gap_min = min(data['gap'])
    gap_std = np.std(data['gap'])

    print(gap_std, gap_max, gap_min)
    #data.to_csv("data/log/high.csv")

    g_withdraw, win_ratio, sharp = get_result(data, max(abs(max_hold), abs(min_hold)))
    print("收益：", data.loc[len(data) - 1, "account"])
    print("最大回撤：", g_withdraw)
    print("胜率：", win_ratio)
    print("sharp比：", sharp)

    # ------------------------------BP-------------------------#
    '''
    #目标
    获取最优网格grids
    根据已有信息，自动调整grids密度，代替人工设置或调整

    #设计思路
    使用逆向传播bp，斜率代替导数

    #注意事项
    实际操作中根据历史价格作出参数调整，因此BP调整参数有效的前提在于，未来价格波动与历史价格波动有一定程度的一致性
    训练速度较慢，实际高频操作中，分为以下几步：
    1.使用human set grids找出相对较好的grids和其他参数，作为初步参数
    2.BP中的初始grids参数，初始化为human set grids里的最优值

    #超参：
        lr学习率
        EPOCH总训练轮数
    '''

    EPOCH = 10  # 3
    lr = 1e-5  # account * 1e-9

    np.random.seed(0)
    # turb = np.random.randn(len(grids))/10
    turb = np.ones(len(grids))

    grids_bp1 = grids[:]
    grids_bp2 = grids[:] + turb
    former_f = grids[:]

    sigma = 2
    grid_max_real = min(max(data['gap']), max(grids) + 2 * gap_std)
    grid_min_real = max(min(data['gap']), min(grids) - 2 * gap_std)
    print(grid_max_real, grid_min_real)
    # grid_max_real = max(data["gap"])
    # grid_min_real = min(data["gap"])

    # to limit bp:
    # i=0,len-1: grid_min, grid_max
    # otherwise: i-1,i+1

    for epoch in range(EPOCH):
        for i in range(len(grids)):
            if epoch == 0:
                x2 = grids_bp2[i]
                x1 = grids_bp1[i]
                data2, points2 = data_process(data_org, grids_bp2, base_line)
                data1, points1 = data_process(data_org, grids_bp1, base_line)
                account_list2 = get_account(data2, points2, grids_bp2, market, service_rate, bail_rate, max_hold,
                                            min_hold, add)
                account_list1 = get_account(data1, points1, grids_bp1, market, service_rate, bail_rate, max_hold,
                                            min_hold, add)
                f2 = account_list2[-1]
                f1 = account_list1[-1]
                delta = (f2 - f1) / (x2 - x1) if x2 != x1 else f2
                tmp = grids_bp2[i] + lr * delta

                # check limit
                if i == 0:
                    if tmp < grid_min_real:
                        grids_bp2[i] = grid_min_real
                    elif tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    else:
                        grids_bp2[i] = tmp
                elif i == len(grids) - 1:
                    if tmp > grid_max_real:
                        grids_bp2[i] = grid_max_real
                    elif tmp < grids_bp2[i - 1]:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp
                else:
                    if tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    elif tmp < grids_bp2[i - 1]:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp

                grids_bp1[i] = x2
                former_f[i] = f2
            else:  # reuse former calculation
                x2 = grids_bp2[i]
                x1 = grids_bp1[i]
                data2, points2 = data_process(data_org, grids_bp2, base_line)
                account_list2 = get_account(data2, points2, grids_bp2, market, service_rate, bail_rate, max_hold,
                                            min_hold, add)
                f2 = account_list2[-1]
                f1 = former_f[i]
                delta = (f2 - f1) / (x2 - x1) if x2 != x1 else f2
                # grids_bp2[i] = grids_bp2[i] + lr * delta
                tmp = grids_bp2[i] + lr * delta

                # check limit
                if i == 0:
                    if tmp < grid_min_real:
                        grids_bp2[i] = grid_min_real
                    elif tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    else:
                        grids_bp2[i] = tmp
                elif i == len(grids) - 1:
                    if tmp > grid_max_real:
                        grids_bp2[i] = grid_max_real
                    elif tmp < grids_bp2[i - 1]:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp
                else:
                    if tmp > grids_bp2[i + 1] - sigma:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    elif tmp < grids_bp2[i - 1] + sigma:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp
                grids_bp1[i] = x2
                former_f[i] = f2

    grids_adjust = nearest_adjust(grids_bp2, grid_max_real, grid_min_real)

    data_adjust, points_adjust = data_process(data_org, grids_adjust, base_line)
    bp_log_adjust = get_account_log(data_adjust, points_adjust, grids_adjust, market, service_rate, bail_rate, max_hold,
                                    min_hold, add)
    account_list_adjust = bp_log_adjust["account"]
    plot_bp(len(data), account_list_adjust, lr, EPOCH, grids_adjust)
    g_withdraw_bp, win_ratio_bp, sharp_bp = get_result(bp_log_adjust, max(abs(max_hold), abs(min_hold)))
    print(grids_adjust)
    print("收益：", bp_log_adjust.loc[len(data) - 1, "account"])
    print("最大回撤：", g_withdraw_bp)
    print("胜率：", win_ratio_bp)
    print("sharp比：", sharp_bp)
    # bp_log_adjust.to_csv("data/log/add.csv")

    return grids_adjust, bp_log_adjust

def train_params2(path,market,service_rate,bail_rate,max_hold,min_hold,base_line,grid_max,grid_min,grid_num,add):
    # ------------------------------human set grids----------------------#
    '''
    这里处理的是人为给定的grids
    使用自动调参时可以不运行
    '''

    grids = np.linspace(grid_min, grid_max, grid_num)
    grids = grids.tolist()
    grids = nearest_adjust(grids, grid_max, grid_min)
    print(grids)


    data_org = load_data(path)
    data, points = data_process(data_org, grids, base_line)
    data = get_account_log(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold,add)
    account_list = data['account']
    plot(len(data), account_list, grid_max, grid_min, grid_num)

    # describe data
    gap_max = max(data['gap'])
    gap_min = min(data['gap'])
    gap_std = np.std(data['gap'])

    print(gap_std, gap_max, gap_min)


    g_withdraw, win_ratio, sharp = get_result(data, max(abs(max_hold),abs(min_hold)))
    print("收益：",data.loc[len(data)-1,"account"])
    print("最大回撤：",g_withdraw)
    print("胜率：",win_ratio)
    print("sharp比：",sharp)


    # ------------------------------BP-------------------------#
    '''
    #目标
    获取最优网格grids
    根据已有信息，自动调整grids密度，代替人工设置或调整

    #设计思路
    使用逆向传播bp，斜率代替导数

    #注意事项
    实际操作中根据历史价格作出参数调整，因此BP调整参数有效的前提在于，未来价格波动与历史价格波动有一定程度的一致性
    训练速度较慢，实际高频操作中，分为以下几步：
    1.使用human set grids找出相对较好的grids和其他参数，作为初步参数
    2.BP中的初始grids参数，初始化为human set grids里的最优值

    #超参：
        lr学习率
        EPOCH总训练轮数
    '''

    EPOCH = 5#3
    lr = 1e-5 # account * 1e-9

    np.random.seed(0)
    # turb = np.random.randn(len(grids))/10
    turb = np.ones(len(grids))

    grids_bp1 = grids[:]
    grids_bp2 = grids[:] + turb
    former_f = grids[:]

    sigma = 2
    grid_max_real = min(max(data['gap']), grid_max+2*gap_std)
    grid_min_real = max(min(data['gap']), grid_min-2*gap_std)
    print(grid_max_real,grid_min_real)
    #grid_max_real = max(data["gap"])
    #grid_min_real = min(data["gap"])

    # to limit bp:
    # i=0,len-1: grid_min, grid_max
    # otherwise: i-1,i+1

    for epoch in range(EPOCH):
        for i in range(len(grids)):
            if epoch == 0:
                x2 = grids_bp2[i]
                x1 = grids_bp1[i]
                data2, points2 = data_process(data_org, grids_bp2, base_line)
                data1, points1 = data_process(data_org, grids_bp1, base_line)
                account_list2 = get_account(data2, points2, grids_bp2, market, service_rate, bail_rate, max_hold,
                                            min_hold, add)
                account_list1 = get_account(data1, points1, grids_bp1, market, service_rate, bail_rate, max_hold,
                                            min_hold, add)
                f2 = account_list2[-1]
                f1 = account_list1[-1]
                delta = (f2 - f1) / (x2 - x1) if x2 != x1 else f2
                tmp = grids_bp2[i] + lr * delta

                # check limit
                if i == 0:
                    if tmp < grid_min_real:
                        grids_bp2[i] = grid_min_real
                    elif tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    else:
                        grids_bp2[i] = tmp
                elif i == len(grids) - 1:
                    if tmp > grid_max_real:
                        grids_bp2[i] = grid_max_real
                    elif tmp < grids_bp2[i - 1]:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp
                else:
                    if tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    elif tmp < grids_bp2[i - 1]:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp

                grids_bp1[i] = x2
                former_f[i] = f2
            else:  # reuse former calculation
                x2 = grids_bp2[i]
                x1 = grids_bp1[i]
                data2, points2 = data_process(data_org, grids_bp2, base_line)
                account_list2 = get_account(data2, points2, grids_bp2, market, service_rate, bail_rate, max_hold,
                                            min_hold, add)
                f2 = account_list2[-1]
                f1 = former_f[i]
                delta = (f2 - f1) / (x2 - x1) if x2 != x1 else f2
                # grids_bp2[i] = grids_bp2[i] + lr * delta
                tmp = grids_bp2[i] +  lr * delta

                # check limit
                if i == 0:
                    if tmp < grid_min_real:
                        grids_bp2[i] = grid_min_real
                    elif tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    else:
                        grids_bp2[i] = tmp
                elif i == len(grids) - 1:
                    if tmp > grid_max_real:
                        grids_bp2[i] = grid_max_real
                    elif tmp < grids_bp2[i - 1]:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp
                else:
                    if tmp > grids_bp2[i + 1]-sigma:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    elif tmp < grids_bp2[i - 1]+sigma:
                        grids_bp2[i] = grids_bp2[i - 1] + sigma
                    else:
                        grids_bp2[i] = tmp
                grids_bp1[i] = x2
                former_f[i] = f2

    grids_adjust = nearest_adjust(grids_bp2, grid_max_real, grid_min_real)

    data_adjust, points_adjust = data_process(data_org, grids_adjust, base_line)
    bp_log_adjust = get_account_log(data_adjust, points_adjust, grids_adjust, market, service_rate, bail_rate, max_hold,
                                      min_hold, add)
    account_list_adjust = bp_log_adjust["account"]
    plot_bp(len(data), account_list_adjust, lr, EPOCH,grids_adjust)
    g_withdraw_bp, win_ratio_bp, sharp_bp = get_result(bp_log_adjust, max(abs(max_hold), abs(min_hold)))
    print(grids_adjust)
    print("收益：", bp_log_adjust.loc[len(data) - 1, "account"])
    print("最大回撤：", g_withdraw_bp)
    print("胜率：", win_ratio_bp)
    print("sharp比：", sharp_bp)
    #bp_log_adjust.to_csv("data/log/add.csv")

    return grids_adjust, bp_log_adjust

def run(path,grids,base_line,market,service_rate,bail_rate,max_hold,min_hold, add):
    data_org = load_data(path)
    data, points = data_process(data_org, grids, base_line)
    data_log = get_account_log(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold, add)
    account_list = data_log["account"]
    plot_run(len(data), account_list, grids)
    g_withdraw, win_ratio, sharp = get_result(data_log, max(abs(max_hold), abs(min_hold)))
    print("收益：", data_log.loc[len(data_log) - 1, "account"])
    print("最大回撤：", g_withdraw)
    print("胜率：", win_ratio)
    print("sharp比：", sharp)
    data_log.to_csv("data/log/d.csv")
    return account_list

def get_result(log, hold):

    account = log["account"]
    idx_h = 0 # 0~i-1最高点的索引
    g_withdraw = float('-inf')# 全局最大回撤 对应最高点和最低点的索引
    pre_account = float('-inf')
    win, total = 0, 0

    for i in range(len(account)):
        # 最大回撤
        if i>0:
            if account[idx_h] < account[i-1]:
                idx_h = i-1

            drawdown = account[idx_h] - account[i]
            if drawdown>g_withdraw:
                g_withdraw = drawdown

            # 胜率
            if log.loc[i,"bag"]!=log.loc[i-1,"bag"]:
                total = total + 1
                if log.loc[i,"account"]>pre_account:
                    win  = win + 1
                pre_account = log.loc[i,"account"]

    win_ratio = win/total

    # sharp
    w = hold*400000
    r_mean = np.mean(account)/w
    r_s = np.std(account,ddof=1)/(w*w)
    r_f = 3.97/100
    sharp = (r_mean-r_f)/r_s

    return g_withdraw, win_ratio, sharp

#%%
def main():

    #------------------------------set params----------------------#
    path = "data/IC_IF/IC/IC2012-2009.csv"
    market = "IC"  # IF IC IH

    # 检验时不需重新设置
    service_rate = 0.000026  # 手续费
    bail_rate = 0
    max_hold = 4 # 仓位上限
    min_hold = -4 # 仓位下限
    base_line = "avg3"  # avg3,avg5:均线选取，需在表中出现该列
    add = 0.4

    # 检验时不需设置
    grid_max =  9# 9 网格上限
    grid_min = -9# 网格下限
    grid_num = 8 # 网格数 固定


# ------------------------------set params----------------------#

    # train
    #grids = [-9, -6.6, -4, -1.4, 1.2, 3.8, 6.4, 9]
    #grids = [-8,-6,-4,-2,0,2,4,6,8]
    #grids = [-9,-7,-5,-3,-1,1,3,5,7]
    #grids = [-7,-5,-3,-1,1,3,5,7,9]
    #grids = [-10,-8,-6, -4, -2, 0, 2, 4, 6, 8]
    #grids, log = train_params(path,market,service_rate,bail_rate,max_hold,min_hold,base_line,grids,add)



    # run
    #grids = [-4.2, -1. ,  2.4,  5. ]
    #grids = np.linspace(grid_min, grid_max, grid_num)
    #grids = grids.tolist()
    #grids = nearest_adjust(grids, grid_max, grid_min)
    #grids = [-5.6, -3.6, -1.8, 0.2, 1.6, 3.6, 5]
    #grids = np.linspace(grid_min,grid_max,grid_num)
    grids = [-9, -6, -3, 0, 3, 6, 9]

    #grids = [-10,-7,-4,-1,2,6,8]
    #grids = [-8,-5,-2,1,4,7,10]
    #grids = [-11,-8,-5,-2,1,4,7]
   # grids = [-7,-4,-1,2,5,8,11]
    account_list = run(path,grids,base_line,market,service_rate,bail_rate,max_hold,min_hold,add)





if __name__ == '__main__':
    main()

