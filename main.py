import pandas as pd
import matplotlib
import sklearn
import sys
import scipy


# %%
# load data
sys.path.insert(0,".")
path = r"data/IF03-12.xlsx"
data = pd.read_excel(path).dropna(axis=0, how='any').reset_index()
data['gap'] = data['spread'] - data['avg3']

# %%

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
    # save in data


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



grids = [-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18]
data['grid'] = data.gap.apply(lambda x: int(find_grid(x,grids)))
grid_log, points = deal_points(data)


# %%
#---------------------------get revenue--------------------------#
def get_change(grid_log):
    # get change in hold at each deal points
    # 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 3,
    change_log = grid_log[:]
    for i in range(len(grid_log)-1,0,-1):
        change_log[i] = change_log[i]-change_log[i-1]
    return change_log[1:]

change_log = get_change(grid_log)

def get_revenue(data, change_log, points):
    for i in range(len(points))：# for each deal



# def service(data, grid_log, points):



'''
def main():
    grids = [-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18]
    data['grid'] = data.gap.apply(lambda x: int(find_grid(x,grids)))


if __name__ == '__main__':
    main()
'''

