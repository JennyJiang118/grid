import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

#%%
def load_data(path):
    sys.path.insert(0,".")
    data = pd.read_excel(path).dropna(axis=0, how='any').reset_index()
    return data


def data_process(data, grids, base_line):
    data['gap'] = data['spread'] - data[base_line]
    data['grid'] = data.gap.apply(lambda x: int(find_grid(x, grids)))
    #grid_log, points = deal_points(data)#
    #change_log = get_change(grid_log)#
    points = cross_points(data)
    #return data, grid_log, points, change_log#
    return data, points


def find_grid(x,grids):
    grid = len(grids)-1
    if x>=grids[0] and x<=grids[-1]:
        for i in range(len(grids) - 1):
            if x >= grids[i] and x < grids[i + 1]:
                grid = i+1
                break
    elif x<grids[0]:
        grid = 1
    return grid


def cross_points(data):
    points = [] # 与网格线相交的点idx
    for i in range(1,len(data)):
        if data.loc[i,"grid"] != data.loc[i-1,"grid"]:
            points.append(i)
    return points



#---------------------------get revenue--------------------------#
#%%
def get_account(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold):
    account_list = [] # bag+revenue-service-bail
    service = 0 # 叠加 每点更新
    bag = 0 # 落袋金额
    bail_stack = [] # 每点push pop并sum
    price_stack = [] # 存储仓内买入、卖空价
    hold = 0 # 总持有数
    cnt = 0 # 网格交叉点计数
    mul = 200 if market == "IC" else 300
    pre_grid = 1 # 上一单所处网格
    for i in range(len(data)):
        if i == points[cnt] and cnt != len(points)-1: # 是交叉点 可能可以交易
            if hold == 0: # 从头开始
                if data.loc[i, "grid"] < (grids[0]+grids[-1])/2: # 所处网格较低，涨空间大，买入
                    hold = 1
                else:
                    hold = -1
                pre_grid = data.loc[i,"grid"]
                service = service + (data.iloc[i,2]+data.iloc[i,3])*mul*service_rate
                bail = (data.iloc[i,2]+data.iloc[i,3])*mul*bail_rate
                bail_stack.append(bail)
                price_stack.append(data.loc[i,"spread"])
            else: # 与前期比较是否需要更改

                # change_hold需要在实际操作中再得出来，否则会有持仓或资金的限制
                intend_change = data.loc[i,"grid"]-pre_grid
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

                if deal:
                    old_hold = hold
                    hold = hold + change_hold
                    pre_grid = data.loc[i,"grid"]
                    service = service + (data.iloc[i, 2] + data.iloc[i, 3]) * mul * service_rate * abs(change_hold)
                    price = data.loc[i,"spread"]
                    # bail & price append
                    bail = (data.iloc[i, 2] + data.iloc[i, 3]) * bail_rate * mul
                    if old_hold * hold > 0:
                        if change_hold * old_hold > 0:  # pay bail
                            for times in range(abs(change_hold)):
                                bail_stack.append(bail)
                                price_stack.append(price)
                        else:  # no pay
                            for times in range(abs(change_hold)):
                                bail_stack.pop()
                                old_price = price_stack.pop()
                                bag = bag + abs(price - old_price)*mul
                    if old_hold * hold == 0:
                        if old_hold == 0:
                            for times in range(abs(change_hold)):
                                bail_stack.append(bail)
                                price_stack.append(price)

                        else:
                            for times in range(abs(change_hold)):
                                bail_stack.pop()
                                old_price = price_stack.pop()
                                bag = bag + abs(price - old_price)*mul
                    else:
                        bag = bag + abs(price*len(price_stack) - sum(price_stack))*mul
                        bail_stack.clear()
                        price_stack.clear()
                        for times in range(abs(hold)):
                            bail_stack.append(bail)
                            price_stack.append(price)


            cnt = cnt + 1

        # i点account计算
        revenue = data.loc[i,"spread"] * hold - sum(price_stack)
        account = revenue + bag - service #- sum(bail_stack)
        account_list.append(account)
    return account_list

def get_revenue(data, change_log, points, market, service_rate, max_hold, min_hold):
    revenue_sum = 0
    revenue_list = []
    cnt = 0
    hold = 0
    for i in range(len(data)):
        if i == points[cnt] and cnt != len(points)-1:
            mul = 200 if market=="IC" else 300
            change_hold = change_log[cnt]

            # hold/bail limit
            deal = False if (hold==max_hold and change_hold==1) or (hold==min_hold and change_hold==-1) else True
            if deal:
                hold = hold + change_hold

                price = data.loc[i,'spread']
                revenue_i = price * mul * change_hold

                service = (data.iloc[i,2]+data.iloc[i,3])*mul*service_rate

                revenue_sum = revenue_sum + revenue_i - service
                cnt = cnt + 1
        revenue_list.append(revenue_sum)
    return revenue_list

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

def plot(len, account_list,grid_max, grid_min, grid_num):
    x = np.linspace(1, len, len)
    #x = data['datetime']
    plt.title("grid max="+str(grid_max)+"  grid min="+str(grid_min)+"  grid num="+str(grid_num))
    plt.plot(x, account_list)
    plt.show()


def plot_bp(len, revenue_list, lr, EPOCH):
    x = np.linspace(1,len,len)
    plt.title("bp revenue "+"lr="+str(lr)+" EPOCH="+str(EPOCH))
    plt.plot(x, revenue_list)
    plt.show()

def plot_run(len, revenue_list, grids):
    x = np.linspace(1, len, len)
    plt.title("grids:"+str(grids))
    plt.plot(x, revenue_list)
    plt.show()

def train_params(path,market,service_rate,bail_rate,max_hold,min_hold,base_line,grid_max,grid_min,grid_num):
    # ------------------------------human set grids----------------------#
    '''
    这里处理的是人为给定的grids
    使用自动调参时可以不运行
    '''
    grids = np.linspace(grid_min, grid_max, grid_num)
    grids = grids.tolist()
    grids = nearest_adjust(grids, grid_max, grid_min)
    data_org = load_data(path)
    # data, grid_log, points, change_log = data_process(data_org,grids,base_line)
    data, points = data_process(data_org, grids, base_line)
    # revenue_list = get_revenue(data, change_log, points, market, service_rate, max_hold, min_hold)
    account_list = get_account(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold)
    plot(len(data), account_list, grid_max, grid_min, grid_num)


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
    EPOCH = 3
    lr = 1e-6  # 1e-6

    np.random.seed(0)
    # turb = np.random.randn(len(grids))/10
    turb = np.ones(len(grids))

    grids_bp1 = grids[:]
    grids_bp2 = grids[:] + turb
    former_f = grids[:]

    # to limit bp:
    # i=0,len-1: grid_min, grid_max
    # otherwise: i-1,i+1
    sigma = int((grid_max - grid_min) / grid_num)
    for epoch in range(EPOCH):
        for i in range(len(grids)):
            if epoch == 0:
                x2 = grids_bp2[i]
                x1 = grids_bp1[i]
                data2, points2 = data_process(data_org, grids_bp2, base_line)
                data1, points1 = data_process(data_org, grids_bp1, base_line)
                account_list2 = get_account(data2, points2, grids_bp2, market, service_rate, bail_rate, max_hold,
                                            min_hold)
                account_list1 = get_account(data1, points1, grids_bp1, market, service_rate, bail_rate, max_hold,
                                            min_hold)
                f2 = account_list2[-1]
                f1 = account_list1[-1]
                delta = (f2 - f1) / (x2 - x1) if x2 != x1 else f2
                tmp = grids_bp2[i] + abs(x2 - x1) * lr * delta  # 动态学习率：x1x2相差小时，delta大

                # check limit
                if i == 0:
                    if tmp < grid_min:
                        grids_bp2[i] = grid_min
                    elif tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    else:
                        grids_bp2[i] = tmp
                elif i == len(grids) - 1:
                    if tmp > grid_max:
                        grids_bp2[i] = grid_max
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
                                            min_hold)
                f2 = account_list2[-1]
                f1 = former_f[i]
                delta = (f2 - f1) / (x2 - x1) if x2 != x1 else f2
                # grids_bp2[i] = grids_bp2[i] + lr * delta
                tmp = grids_bp2[i] + abs(x2 - x1) * lr * delta

                # check limit
                if i == 0:
                    if tmp < grid_min:
                        grids_bp2[i] = grid_min
                    elif tmp > grids_bp2[i + 1]:
                        grids_bp2[i] = grids_bp2[i + 1] - sigma
                    else:
                        grids_bp2[i] = tmp
                elif i == len(grids) - 1:
                    if tmp > grid_max:
                        grids_bp2[i] = grid_max
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
    data_bp, points_bp = data_process(data_org, grids_bp2, base_line)
    # account_list_bp = get_account(data_bp, points_bp, grids_bp2, market, service_rate,bail_rate, max_hold, min_hold)
    # plot_bp(data, account_list_bp,lr,EPOCH)
    # print(grids_bp2)

    grids_adjust = nearest_adjust(grids_bp2, grid_max, grid_min)
    account_list_adjust = get_account(data_bp, points_bp, grids_adjust, market, service_rate, bail_rate, max_hold,
                                      min_hold)
    plot_bp(len(data), account_list_adjust, lr, EPOCH)
    return grids_adjust

def run(path,grids,base_line,market,service_rate,bail_rate,max_hold,min_hold):
    data_org = load_data(path)
    data, points = data_process(data_org, grids, base_line)
    account_list = get_account(data, points, grids, market, service_rate, bail_rate, max_hold, min_hold)
    plot_run(len(data), account_list, grids)
    return account_list

def main():
#%%
    #------------------------------set params----------------------#
    path = "data/IF03-01.xlsx"
    market = "IF"  # IF IC IH

    # 检验时不需重新设置
    service_rate = 0.000026  # 手续费
    bail_rate = 0.14    # 保证金
    max_hold = 4 # 仓位上限
    min_hold = -4 # 仓位下限
    base_line = "avg3"  # avg3,avg5:均线选取，需在表中出现该列

    # 检验时不需设置
    grid_max = 3 # 网格上限
    grid_min = -3 # 网格下限
    grid_num = 4 # 网格数 固定

# ------------------------------set params----------------------#

    # train
    grids = train_params(path,market,service_rate,bail_rate,max_hold,min_hold,base_line,grid_max,grid_min,grid_num)

    # run
    #grids = [-4.2, -1. ,  2.4,  5. ]
    #account = run(path,grids,base_line,market,service_rate,bail_rate,max_hold,min_hold)





if __name__ == '__main__':
    main()

