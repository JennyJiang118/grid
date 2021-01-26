import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
import copy

#%%
def load_data(path):
    sys.path.insert(0,".")
    data = pd.read_excel(path).dropna(axis=0, how='any').reset_index()
    return data


def data_process(data, grids, base_line):
    data['gap'] = data['spread'] - data[base_line]
    data['grid'] = data.gap.apply(lambda x: int(find_grid(x, grids)))
    grid_log, points = deal_points(data)
    change_log = get_change(grid_log)
    return data, grid_log, points, change_log



def find_grid(x, grids):
    zero_idx = -1
    for i,item in enumerate(grids):
        if item == 0:
            zero_idx = i
            break

    grid = len(grids)-zero_idx-1 if x>0 else -zero_idx

    if x>grids[0] and x<grids[-1]: # within price limit
        for i in range(len(grids)):
            if grids[i] >= x:
                grid = i-zero_idx-1
                break
    return grid


def deal_points(data):
    # which grid
    grid_log = [] # real grids change log
    points = []

    # initial starting point
    grid_log.append(data.loc[0,'grid'])

    # get grid_log & points
    for i in range(1,len(data)):
        # grid update?
        if data.loc[i,'grid'] != data.loc[i-1,'grid']:
            grid_log.append(data.loc[i,'grid'])
            points.append(i)
    return grid_log, points


#---------------------------get revenue--------------------------#
def get_change(grid_log):
    # get change in hold at each deal points
    change_log = grid_log[:]
    for i in range(len(grid_log)-1,0,-1):
        change_log[i] = change_log[i]-change_log[i-1]
    return change_log[1:]


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

def plot(data, revenue_list,grid_max, grid_min, grid_num):
    x = np.linspace(1, len(data), len(data))
    #x = data['datetime']
    plt.title("grid max="+str(grid_max)+"  grid min="+str(grid_min)+"  grid num="+str(grid_num))
    plt.plot(x, revenue_list)
    plt.show()

def plot_bp(data, revenue_list, lr, EPOCH):
    x = np.linspace(1,len(data),len(data))
    plt.title("bp revenue "+"lr="+str(lr)+" EPOCH="+str(EPOCH))
    plt.plot(x, revenue_list)
    plt.show()

def main():
#%%
    #------------------------------set params----------------------#
    path = "data/IC03-12.xlsx"
    market = "IC"  # IF IC IH
    service_rate = 0.000026  # 手续费
    max_hold = 4 # 仓位上限
    min_hold = -4 # 仓位下限
    base_line = "avg5"  # avg3,avg5:均线选取，需在表中出现该列
    grid_max = 8 # 网格上限
    grid_min = -8 # 网格下限
    grid_num = 5 # 网格数

#------------------------------human set grids----------------------#
    '''
    这里处理的是人为给定的grids
    使用自动调参时可以不运行
    '''
    grids = np.linspace(grid_min, grid_max, grid_num)
    grids = grids.tolist()
    data_org = load_data(path)
    data, grid_log, points, change_log = data_process(data_org,grids,base_line)
    revenue_list = get_revenue(data, change_log, points, market, service_rate, max_hold, min_hold)
    plot(data, revenue_list,grid_max,grid_min,grid_num)

#%%
    #------------------------------BP-------------------------#
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
    2.实际操作时，根据对处理速度的要求，减少训练轮数EPOCH
    3.BP中的初始grids参数，初始化为human set grids里的最优值
    
    #超参：
        lr学习率
        EPOCH总训练轮数
    '''
    EPOCH = 3
    lr = 1e-6

    np.random.seed(0)
    #turb = np.random.randn(len(grids))/10
    turb = np.ones(len(grids))


    grids_bp1 = grids[:]
    grids_bp2 = grids[:] + turb
    former_f = grids[:]

    '''
    grids_bp1 = copy.deepcopy(grids)
    grids_bp2 = copy.deepcopy(grids) + turb
    former_f = copy.deepcopy(grids)
    '''

    # to limit bp:
    # i=0,len-1: grid_min, grid_max
    # otherwise: i-1,i+1
    sigma = 1
    for epoch in range(EPOCH):
        for i in range(len(grids)):
            if epoch==0:
                x2 = grids_bp2[i]
                x1 = grids_bp1[i]
                data2, grid_log2, points2, change_log2 = data_process(data_org,grids_bp2,base_line)
                data1, grid_log1, points1, change_log1 = data_process(data_org,grids_bp1,base_line)
                revenue_list2 = get_revenue(data2, change_log2, points2, market, service_rate, max_hold, min_hold)
                revenue_list1 = get_revenue(data1, change_log1, points1, market, service_rate, max_hold, min_hold)
                f2 = revenue_list2[-1]
                f1 = revenue_list1[-1]
                delta = (f2-f1)/(x2-x1) if x2!=x1 else f2
                tmp = grids_bp2[i] + lr*delta


                # check limit
                if i==0:
                    if tmp<grid_min:
                        grids_bp2[i]=grid_min
                    elif tmp>grids_bp2[i+1]:
                        grids_bp2[i]=grids_bp2[i+1]-sigma
                    else:
                        grids_bp2[i]=tmp
                elif i==len(grids)-1:
                    if tmp>grid_max:
                        grids_bp2[i]=grid_max
                    elif tmp<grids_bp2[i-1]:
                        grids_bp2[i]=grids_bp2[i-1]+sigma
                    else:
                        grids_bp2[i]=tmp
                else:
                    if tmp>grids_bp2[i+1]:
                        grids_bp2[i]=grids_bp2[i+1]-sigma
                    elif tmp<grids_bp2[i-1]:
                        grids_bp2[i]=grids_bp2[i-1]+sigma
                    else:
                        grids_bp2[i]=tmp


                grids_bp1[i] = x2
                former_f[i] = f2
            else:   #reuse former calculation
                x2 = grids_bp2[i]
                x1 = grids_bp1[i]
                data2, grid_log2, points2, change_log2 = data_process(data_org,grids_bp2,base_line)
                revenue_list2 = get_revenue(data2, change_log2, points2, market, service_rate, max_hold, min_hold)
                f2 = revenue_list2[-1]
                f1 = former_f[i]
                delta = (f2 - f1) / (x2 - x1) if x2!=x1 else f2
                #grids_bp2[i] = grids_bp2[i] + lr * delta
                tmp = grids_bp2[i] + lr*delta

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
    #grids_bp2 = grids_bp2 - grids_bp2[int(len(grids_bp2)/2)]
    data_bp, grid_log_bp, points_bp, change_log_bp = data_process(data_org,grids_bp2,base_line)
    revenue_list_bp = get_revenue(data_bp, change_log_bp, points_bp, market, service_rate,max_hold, min_hold)
    plot_bp(data, revenue_list_bp,lr,EPOCH)
    print(grids_bp2)


if __name__ == '__main__':
    main()

