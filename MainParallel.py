#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/02/15
# @Author  :
# @Site    :
# @File    : MainParallel.py
# @Software: PyCharm

import math
import numpy as np
import pandas as pd
import datetime
from ModelFunctionParallel20230215 import forwardInNCBLParallel, forwardInCBLParallel, \
    backwardInNCBLParallel, backwardInCBLParallel, transferCoord, determineScenario
import sys
import time
import os

'''用于保存打印日志，可忽略'''
class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        # self.log = open(fileN, "a")
        self.log = open(fileN, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

'''1、读入污染源文件名'''
def loadEmissionSource(emission_source_path, expr_id, alpha):
    # 'Date_start', 'Date_end', 'x', 'y', 'z', 'Q_s'
    '''
    expr_id: 实验id,一共68组
    '''
    df = pd.read_csv(emission_source_path, encoding='utf-8')
    target_df = df[df['Experiment number'] == expr_id]
    # 弧上受体数/采样点数，源x坐标，源y坐标，源强，源高
    target_df_new = target_df[['Date', 'Time(start and stop times)',
                               'X-coordinate of source', 'Y-coordinate of source',
                               'Tracer release rate', 'Tracer release height']]
    # target_df_new.rename(columns={'X-coordinate of source': 'x_lgc', 'Y-coordinate of source': 'y_lgc',
    #                           'Tracer release rate': 'Qs', 'Tracer release height': 'z'}, inplace=True)
    target_df_new.columns = ['Date', 'Time(start and stop times)', 'x_lgc', 'y_lgc', 'Qs', 'z']
    '''重构坐标'''
    x_wc_list = []
    y_wc_list = []
    X_coor_array = target_df_new['x_lgc'].values
    Y_coor_array = target_df_new['y_lgc'].values
    for X_coor, Y_coor in zip(X_coor_array, Y_coor_array):
        x_wc, y_wc = transferCoord(X_coor, Y_coor, alpha)
        x_wc_list.append(x_wc)
        y_wc_list.append(y_wc)
    target_df_new = target_df_new.copy()
    target_df_new['x_wc'] = np.array(x_wc_list)
    target_df_new['y_wc'] = np.array(y_wc_list)
    '''重构时间'''
    # print(target_df_new)
    date_array = target_df_new['Date'].values
    time_duration_array = target_df_new['Time(start and stop times)'].values
    date_start_list = []
    date_end_list = []
    for date, time in zip(date_array, time_duration_array):
        temp_list = str(date).split('/')
        # print(temp_list)
        M, D, Y = int(temp_list[0]), int(temp_list[1]), 1900+int(temp_list[2])
        temp_list = str(time).split('-')
        # print(temp_list)
        H_start, Min_start= int(temp_list[0][:-2]), int(temp_list[0][-2:])
        H_end, Min_end = int(temp_list[1][:-2]), int(temp_list[1][-2:])
        date_start_list.append(pd.to_datetime(datetime.datetime(Y, M, D, H_start, Min_start)))
        date_end_list.append(pd.to_datetime(datetime.datetime(Y, M, D, H_end, Min_end)))
    target_df_new = target_df_new.copy()
    target_df_new['Date_start'] = np.array(date_start_list)
    target_df_new['Date_end'] = np.array(date_end_list)
    target_df_new['Rs'] = np.zeros(len(target_df_new))
    final_df = target_df_new[['Date_start', 'Date_end', 'x_lgc', 'y_lgc', 'x_wc', 'y_wc', 'z', 'Rs', 'Qs']]
    final_df = final_df.head(1)  # 目前仅有一个排放源
    # final_df = final_df.ix[0]
    # print(final_df)
    return final_df

'''采样点高度为1.5m,有浓度观测值'''
def loadSamplingPoint(expr_id, alpha, z=1.5):
    # 'x', 'y', 'z', 'C'
    df_list = []
    for i in range(1, 6):  # 5道弧,存储第几道弧信息1128
        # X, Y, observed concentration(mg/m3)
        sample_point_path = './相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_obsinf_E'+str(expr_id)+'A'+str(i)+'.csv'
        df = pd.read_csv(sample_point_path, encoding='utf-8')
        df['ArcID'] = (np.ones(len(df))*i).astype(int)
        df_list.append(df)
    total_df = pd.concat(df_list, ignore_index=True)
    total_df['z'] = np.ones(len(total_df))*z
    X_coor_array = total_df['X-coordinate'].values
    Y_coor_array = total_df['Y-coordinate'].values
    x_lgc_list = []
    y_lgc_list = []
    x_wc_list = []
    y_wc_list = []
    for X_coor, Y_coor in zip(X_coor_array, Y_coor_array):
        x_lgc = -math.cos(math.radians(X_coor))*Y_coor
        '''
        if X_coor <= 90:
            x_lgc = -math.cos(math.radians(X_coor))*Y_coor
            # x_lgc = -math.cos(X_coor)*Y_coor
        else:
            x_lgc = math.cos(math.radians(X_coor))*Y_coor
            # x_lgc = math.cos(X_coor)*Y_coor
        '''
        y_lgc = math.sin(math.radians(X_coor))*Y_coor
        x_wc, y_wc = transferCoord(x_lgc, y_lgc, alpha)
        x_lgc_list.append(x_lgc)
        y_lgc_list.append(y_lgc)
        x_wc_list.append(x_wc)
        y_wc_list.append(y_wc)

    total_df['x_lgc'] = np.array(x_lgc_list)
    total_df['y_lgc'] = np.array(y_lgc_list)
    total_df['x_wc'] = np.array(x_wc_list)
    total_df['y_wc'] = np.array(y_wc_list)
    total_df['WD'] = np.ones(len(x_lgc_list))*alpha

    '''需进行坐标转换，将极坐标转化为平面坐标'''
    result_df = total_df[['X-coordinate', 'Y-coordinate', 'x_lgc', 'y_lgc', 'x_wc', 'y_wc', 'z', 'WD', 'ArcID', 'Observed Concentration']]
    result_df.rename(columns={'Observed Concentration':'C'}, inplace=True)
    result_df['X-coordinate'] = result_df['X-coordinate'].astype(int)
    result_df['Y-coordinate'] = result_df['Y-coordinate'].astype(int)
    # print(result_df)
    return result_df

def loadAllSamplerPoint(alpha):
    X_coordinate_arc = list(range(0, 181, 2))
    # Y_coordinate = [50, 100, 200, 400]
    X_coordinate_arc5 = list(range(0, 181, 1))

    point_coor_1 = [[x, 50, 1.5, 1] for x in X_coordinate_arc]
    point_coor_2 = [[x, 100, 1.5, 2] for x in X_coordinate_arc]
    point_coor_3 = [[x, 200, 1.5, 3] for x in X_coordinate_arc]
    point_coor_4 = [[x, 400, 1.5, 4] for x in X_coordinate_arc]

    point_coor_5 = [[x, 800, 1.5, 5] for x in X_coordinate_arc5]
    result_list = point_coor_1+point_coor_2+point_coor_3+point_coor_4+point_coor_5
    result_array = np.array(result_list) #'X-coordinate', 'Y-coordinate', z, 'ArcID'

    coor_list = []
    for X_coor, Y_coor in zip(result_array[:, 0], result_array[:, 1]):
        x_lgc = -math.cos(math.radians(X_coor)) * Y_coor
        y_lgc = math.sin(math.radians(X_coor)) * Y_coor
        x_wc, y_wc = transferCoord(x_lgc, y_lgc, alpha)
        temp_list = [x_lgc, y_lgc, x_wc, y_wc]
        coor_list.append(temp_list)
    coor_array = np.array(coor_list)  # x_wc, y_wc

    total_array = np.concatenate((result_array, coor_array), axis=1)
    # print(total_array)
    df = pd.DataFrame(total_array, columns=['X-coordinate', 'Y-coordinate', 'z', 'ArcID', 'x_lgc', 'y_lgc', 'x_wc', 'y_wc'])
    df['ArcID'] = df['ArcID'].astype(int)
    df['X-coordinate'] = df['X-coordinate'].astype(int)
    df['Y-coordinate'] = df['Y-coordinate'].astype(int)
    # print(df)
    return df


def main(emiss_df, sampling_point_df, mete_list, N, start_time, dt_min, mode_type, scenario_type,
         traj_file_path, conc_file_path, count_file_path,
         conc_file_path_b1, conc_file_path_b2, conc_file_path_b3, process_nums):
    # print('emiss_df:')
    # print(emiss_df)
    # print(emiss_df['Qs'])
    if mode_type=='forward':
        if scenario_type=='NCBL':
            forwardInNCBLParallel(emission_source_df=emiss_df, sampling_point_df=sampling_point_df,
                                  meteoro_list=mete_list, N=N, start_time=start_time, dt_min=dt_min,
                                  traj_file_path=traj_file_path, conc_file_path=conc_file_path,
                                  count_file_path=count_file_path, process_nums=process_nums)
        else:
            forwardInCBLParallel(emission_source_df=emiss_df, sampling_point_df=sampling_point_df,
                                 meteoro_list=mete_list, N=N, start_time=start_time, dt_min=dt_min,
                                 traj_file_path=traj_file_path, conc_file_path=conc_file_path,
                                 count_file_path=count_file_path, process_nums=process_nums)
    else: # mode_type=='backward'
        if scenario_type=='NCBL':
            backwardInNCBLParallel(sampling_point_df=sampling_point_df, emiss_df=emiss_df,
                                   meteoro_list=mete_list, N=N, start_time=start_time, dt_min=dt_min,
                                   traj_file_path=traj_file_path, conc_file_path=conc_file_path_b1,
                                   count_file_path=count_file_path, process_nums=process_nums)

        else:
            backwardInCBLParallel(sampling_point_df=sampling_point_df, emiss_df=emiss_df,
                                  meteoro_list=mete_list, N=N, start_time=start_time, dt_min=dt_min,
                                  traj_file_path=traj_file_path, conc_file_path=conc_file_path_b1,
                                  count_file_path=count_file_path, process_nums=process_nums)


    return


def loadParaData(expr_id):
    df = pd.read_csv('草场试验参数记录(去重).csv', encoding='utf-8-sig')
    target_df = df[df['Experiment number']==expr_id]
    print(target_df.columns)
    expr_time = target_df[['Time(start and stop times)']]
    Min = int(str(expr_time)[-7:-5])
    # print(target_df[['u*', 'w*', 'Zic', 'Zim','L', 'z0',  'WD', 'secn_type','Y', 'M', 'D', 'HR']].values)
    u_star, w_star, Zic, Zim, L, z_0, alpha, secn_type, Y, M, D, HR = list(target_df[['u*', 'w*', 'Zic', 'Zim',
                                                                                 'L', 'z0',  'WD', 'secn_type',
                                                                                 'Y', 'M', 'D', 'HR']].values[0])
    Z_i = max(Zic, Zim)
    return u_star, w_star, Z_i, L, z_0, alpha, secn_type, int('19'+str(int(Y))), int(M), int(D), int(HR), Min

def correctWD(sampling_point_df):
    arcid_list = set(list(sampling_point_df['ArcID'].values))
    wd_list = []
    for id in arcid_list:
        target_df = sampling_point_df[sampling_point_df['ArcID']==id]
        temp_df = target_df.loc[target_df['C'].idxmax()]
        wd_list.append(temp_df['X-coordinate'])
    # print(wd_list)
    alpha = sum(wd_list)/len(wd_list)+90
    return alpha

if __name__ == '__main__':
    '''保存输出日志'''
    # sys.stdout = Logger('exprid43_log_0111.txt')

    '''--------------第18组实验(NCBL)---------------'''
    '''
    expr_id = 18
    N = 1000
    date = '20230606'
    result_folder = './grassBackwardResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C', 'WD']]  # 带有观测浓度值的采样点
    alpha_corr = correctWD(sampling_point_df)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha_corr]
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha_corr)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1

    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]
    print(target_sample_point_df)
    # 1-4道弧增加4个采样点，5道弧增加4
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i != 5:
            x_add_list = [x_array_min - 2 * j for j in range(1, 5)] + [x_array_max + 2 * j for j in range(1, 5)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 5)] + [x_array_max + j for j in range(1, 5)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = one_data.values[0]
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    # print(conc_sampling_point_df)
    # print(all_sampling_point_df)

    # exit()
    # sampling_point_df = # 在有观测的基础上增加5度左右的区域
    # from ResultAnalyze import plotSamplerDistribution
    # plotSamplerDistribution(sampling_point_df, fig_path=True)
    # sampling_point_df.to_csv('exprid' + str(expr_id) + '_samplinginf.csv')
    # print(len(sampling_point_df))
    # print(sampling_point_df[['x_wc', 'y_wc', 'x_lgc', 'y_lgc']])
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='backward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=64)
    end = time.time()
    print('total run time:', end - start)
    exit()
    # '''
    '''--------------第18组实验(NCBL)---------------'''

    '''
    # 17组 NCBL
    expr_id = 17
    date = '20230606'
    N = 1000
    result_folder = './grassBackwardResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C', 'WD']]  # 带有观测浓度值的采样点
    alpha_corr = correctWD(sampling_point_df)
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha_corr)
    print('mete_list:', mete_list)
    print('old alpha:', alpha, 'alpha:', alpha_corr)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha_corr]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha_corr)
    start = time.time()
    main(emiss_df, sampling_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='backward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=48)
    end = time.time()
    exit()  # '''

    '''--------------第21组实验(NCBL)---------------'''
    # '''存在一些问题
    expr_id = 21
    N = 1000
    date = '20230606'
    result_folder = './grassBackwardResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    dt_min = 0.1
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C', 'WD']]  # 带有观测浓度值的采样点
    alpha_corr = correctWD(sampling_point_df)
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha_corr)
    print('old alpha:', alpha, 'alpha:', alpha_corr)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha_corr]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1

    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    # print(emiss_df)
    print(emiss_df[['x_wc', 'y_wc', 'x_lgc', 'y_lgc']])

    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点
    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]
    print(target_sample_point_df)

    # 未校正风向，前三道弧左侧增加4,6,8个采样点，第四道弧左侧增加10个采样点，第五道弧左侧增加12个采样点
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        # 未校正风向
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 5)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 13)]

        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = one_data.values[0]
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)

    # sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])
    # sampling_point_df.to_csv('exprid'+str(expr_id)+'_samplinginf.csv')
    # print(len(sampling_point_df))
    # print(sampling_point_df[['x_wc', 'y_wc', 'x_lgc', 'y_lgc']])
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='backward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()
    # '''
    '''--------------第21组实验(NCBL)---------------'''


    '''--------------第23组实验(NCBL)---------------'''
    # '''存在一些问题
    expr_id = 23
    N = 1000
    date = '20230606'
    result_folder = './grassBackwardResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C', 'WD']]  # 带有观测浓度值的采样点
    alpha_corr = correctWD(sampling_point_df)
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha_corr)
    print('mete_list:', mete_list)
    print('old alpha:', alpha, 'alpha:', alpha_corr)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha_corr]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    start = time.time()
    main(emiss_df, sampling_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='backward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()








    '''20230327进行后向计算'''
    # 7组 CBL
    expr_id = 7
    date = '20230508'
    N = 1
    result_folder = './grassBackwardResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C', 'WD']]  # 带有观测浓度值的采样点
    alpha_corr = correctWD(sampling_point_df)
    sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha_corr)
    print('mete_list:', mete_list)
    print('old alpha:', alpha, 'alpha:', alpha_corr)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha_corr]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha_corr)
    start = time.time()
    main(emiss_df, sampling_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='backward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=4)
    end = time.time()
    exit()














    '''对流情形使用NCBL模型'''
    '''--------------第62组实验(NCBL)---------------'''
    expr_id = 62
    N = 4
    date = '20230328'
    result_folder = './grassBackwardResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    Scenario = 'NCBL'
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加15个采样点，2-3道弧增加15、15个采样点，4-5道弧增加15、15
    # NCBL 1道弧增加15个采样点，2-3道弧增加15、15个采样点，4-5道弧增加15、25
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 26)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=4)
    end = time.time()
    print('total run time:', end - start)

    exit()



    '''--------------第57组实验(CBL)---------------'''
    expr_id = 57
    N = 20000
    date = '20230325'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    Scenario = 'NCBL'
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加8个采样点，2-3道弧增加15、15个采样点，4-5道弧增加15、15
    # NCBL 1道弧增加8个采样点，2-3道弧增加15、15个采样点，4-5道弧增加15、25
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 8)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 26)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()



    '''--------------第51组实验(NCBL)---------------'''
    expr_id = 51
    N = 20000
    date = '20230325'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    Scenario = 'NCBL'
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # CBL 1道弧增加1个采样点，2-3道弧增加10、15个采样点，4-5道弧增加15、15
    # NCBL 1道弧增加1个采样点，2-3道弧增加10、15个采样点，4-5道弧增加15、25
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 2)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 26)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()

    '''--------------第50组实验(NCBL)---------------'''
    expr_id = 50
    N = 20000
    date = '20230325'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    Scenario = 'NCBL'
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # CBL: 1道弧增加10个采样点，2-3道弧增加10、15个采样点，4-5道弧增加15、15
    # NCBL:1道弧增加10个采样点，2-3道弧增加10、15个采样点，4-5道弧增加15、25
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 26)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()



    '''--------------第45组实验(NCBL)---------------'''
    expr_id = 45
    N = 20000
    date = '20230325'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    Scenario = 'NCBL'
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加8个采样点，2-3道弧增加8、8个采样点，4-5道弧增加15、15
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 16)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()

    '''--------------expr44(NCBL)-------------------'''
    expr_id = 44
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha)[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点
    print(conc_sampling_point_df)
    start = time.time()
    main(emiss_df, conc_sampling_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''



    '''--------------expr43(NCBL)-------------------'''
    expr_id = 43
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha)[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点
    print(conc_sampling_point_df)
    start = time.time()
    main(emiss_df, conc_sampling_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''

    '''--------------第33组实验(NCBL)---------------'''
    expr_id = 33
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加8个采样点，2-3道弧增加8、8个采样点，4-5道弧增加15、15
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 16)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()




    '''--------------expr25(NCBL)-------------------'''
    expr_id = 25
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha)[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点
    print(conc_sampling_point_df)
    start = time.time()
    main(emiss_df, conc_sampling_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''




    '''--------------expr19(NCBL)-------------------'''
    expr_id = 19
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 左侧 1道弧增加2个采样点，2-3道弧增加6、6个采样点，4-5道弧增加10、10
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 3)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 11)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)

    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''



    '''--------------expr16(NCBL)-------------------'''
    expr_id = 16
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 右侧 1道弧增加2个采样点，2-3道弧增加4、4个采样点，4-5道弧增加10、10
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_max + 2 * j for j in range(1, 3)]
        elif i == 2:
            x_add_list = [x_array_max + 2 * j for j in range(1, 5)]
        elif i == 3:
            x_add_list = [x_array_max + 2 * j for j in range(1, 5)]
        elif i == 4:
            x_add_list = [x_array_max + 2 * j for j in range(1, 11)]
        else:
            x_add_list = [x_array_max + j for j in range(1, 11)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)

    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''


    '''--------------第15组实验(NCBL)---------------'''
    expr_id = 15
    N = 20000
    date = '20230324'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    Scenario = 'NCBL'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加2个采样点，2-3道弧增加6、8个采样点，4-5道弧增加15、10
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 3)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 11)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------第10组实验(NCBL)---------------'''
    expr_id = 10
    N = 20000
    date = '20230324'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    Scenario = 'NCBL'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加10个采样点，2-3道弧增加15、15个采样点，4-5道弧增加15、25
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 26)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------expr9(NCBL)-------------------'''
    expr_id = 9
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha)[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点
    print(conc_sampling_point_df)
    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加8个采样点，2-3道弧增加10、12个采样点，4-5道弧增加12、12
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 13)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''





    '''--------------expr8(NCBL)-------------------'''
    expr_id = 8
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha)[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点
    print(conc_sampling_point_df)
    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加8个采样点，2-3道弧增加10、12个采样点，4-5道弧增加12、20
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 21)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(
             N) + '_' + date + '.csv',
         conc_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_conc_' + date + '.txt',
         count_file_path=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder + Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''



    '''--------------expr7(NCBL)-------------------'''
    expr_id = 7
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min= loadParaData(expr_id=expr_id)
    Scenario = 'NCBL'
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 20000
    date = '20230324'
    result_folder = './grassResult/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha)[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点
    print(conc_sampling_point_df)
    start = time.time()
    main(emiss_df, conc_sampling_point_df, mete_list,
         N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=result_folder+Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''


    # '''
    expr_id = 23
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min= loadParaData(expr_id=expr_id)
    # Scenario = 'NCBL'
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    N = 100
    date = '20230323'
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    print(start_time)
    print('Scenario:', Scenario)
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=alpha)
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=alpha)[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点
    print(conc_sampling_point_df)
    start = time.time()
    main(emiss_df, conc_sampling_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=32)
    end = time.time()
    print('total run time:', end - start)
    exit() # '''

    '''--------------第54组实验(NCBL)---------------'''
    expr_id = 54
    N = 20000
    date = '20230317add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧（右侧）增加6个采样点，2-3道弧增加8、10个采样点，4-5道弧增加12、14
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_max + 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_max + 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_max + 2 * j for j in range(1, 11)]
        elif i == 4:
            x_add_list = [x_array_max + 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_max + j for j in range(1, 15)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=32)
    end = time.time()
    print('total run time:', end - start)

    exit()
















    '''--------------第8组实验(CBL)---------------'''
    expr_id = 8
    N = 20000
    date = '20230318add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加10个采样点，2-3道弧增加10、15个采样点，4-5道弧增加15、15
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 21)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()



    '''--------------第59组实验(NCBL)---------------'''
    expr_id = 59
    N = 20000
    date = '20230317add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6+2个采样点，2-3道弧增加8+2、10+2个采样点，4-5道弧增加12、14+6
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 21)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()




    '''--------------第58组实验(NCBL)---------------'''
    expr_id = 58
    N = 20000
    date = '20230317add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加8+5个采样点，2-3道弧增加10+5、12+2个采样点，4-5道弧增加14+2、16+16
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 14)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 16)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 15)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 17)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 33)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)
    exit()




    '''--------------第48组实验(NCBL)---------------'''
    expr_id = 48
    N = 20000
    date = '20230317add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6个采样点，2-3道弧增加8、10个采样点，4-5道弧增加12、14+8
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 23)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------第55组实验(NCBL)---------------'''
    expr_id = 55
    N = 20000
    date = '20230315add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6个采样点，2-3道弧增加8、10个采样点，4-5道弧增加12、14
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 15)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()





    '''--------------第53组实验(NCBL)---------------'''
    expr_id = 53
    N = 20000
    date = '20230315add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6个采样点，2-3道弧增加8、10个采样点，4-5道弧增加12、14
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 15)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()

    '''--------------第42组实验(NCBL)---------------'''
    expr_id = 42
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6个采样点，2-3道弧增加8、10个采样点，4-5道弧增加12、14
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 15)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------第41组实验(NCBL)---------------'''
    expr_id = 41
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6个采样点，2-3道弧增加8、10个采样点，4-5道弧增加12、14
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 15)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------第38组实验(NCBL)---------------'''
    expr_id = 38
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6个采样点，2-3道弧增加8、10个采样点，4-5道弧增加12、14
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 11)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 13)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 15)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------第37组实验(NCBL)---------------'''
    expr_id = 37
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加6个采样点，2-3道弧增加5、7个采样点，4-5道弧增加8、12
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 6)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 8)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 13)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------第36组实验(NCBL)---------------'''
    expr_id = 36
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加1个采样点，2-3道弧增加1、4个采样点，4-5道弧增加8、12
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 2)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 2)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 5)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 9)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 13)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()



    '''--------------第34组实验(NCBL)---------------'''
    expr_id = 34
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加2个采样点，2-3道弧增加4、6个采样点，4-5道弧增加6、8
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 3)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 5)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 9)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()


    '''--------------第32组实验(NCBL)---------------'''
    expr_id = 32
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min = loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加1个采样点，2-3道弧增加4、6个采样点，4-5道弧增加6、8
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 2)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 5)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 9)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)

    exit()




    '''--------------第29组实验(NCBL)---------------'''
    expr_id = 29
    N = 20000
    date = '20230314add'
    u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min= loadParaData(expr_id=expr_id)
    Scenario = determineScenario(Z_i, L, u_star, w_star)
    print(u_star, w_star, Z_i, L, z_0, alpha, Scenario, Y, M, D, HR, Min)
    mete_list = [L, u_star, w_star, Z_i, z_0, alpha]
    start_time = pd.to_datetime(datetime.datetime(Y, M, D, HR, Min))
    dt_min = 0.1
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df)
    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]  # 带有观测浓度值的采样点

    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z', 'C']]
    print(target_sample_point_df)
    # 1道弧增加1个采样点，2-3道弧增加4个采样点，4-5道弧增加6
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        if i == 1:
            x_add_list = [x_array_min - 2 * j for j in range(1, 2)]
        elif i == 2:
            x_add_list = [x_array_min - 2 * j for j in range(1, 5)]
        elif i == 3:
            x_add_list = [x_array_min - 2 * j for j in range(1, 5)]
        elif i == 4:
            x_add_list = [x_array_min - 2 * j for j in range(1, 7)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 7)]
        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = np.append(one_data.values[0], 0)
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=16)
    end = time.time()
    print('total run time:', end - start)



    exit()










    '''--------------第17组实验(NCBL)---------------'''
    '''
    expr_id = 17
    N = 4
    date = '20230216'
    start_time = pd.to_datetime(datetime.datetime(1956, 7, 23, 20, 00))
    dt_min = 0.1
    mete_list = [103.1, 0.222, np.nan, 241, 0.006, 172]
    # mete_list = [103.1, 0.222, np.nan, 241, 0.006, 188]  # 校正风向
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    emiss_df = loadEmissionSource(emission_source_path='./相关文档/PrairieGrass/PGARCS.DAT解析/PGARCS.DAT_basicinf.csv',
                                  expr_id=expr_id, alpha=mete_list[-1])
    print(emiss_df[['x_wc', 'y_wc', 'x_lgc', 'y_lgc']])
    # sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])

    conc_sampling_point_df = loadSamplingPoint(expr_id=expr_id, alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 带有观测浓度值的采样点
    all_sampling_point_df = loadAllSamplerPoint(alpha=mete_list[-1])[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]  # 全部的采样点
    target_sample_point_df = conc_sampling_point_df[
        ['ArcID', 'X-coordinate', 'Y-coordinate', 'x_wc', 'y_wc', 'x_lgc', 'y_lgc', 'z']]
    print(target_sample_point_df)
    # 未校正风向，前三道弧左侧增加5个采样点，第四道弧左侧增加8个采样点，第五道弧左侧增加12个采样点
    arcid_list = [1, 2, 3, 4, 5]
    num = len(conc_sampling_point_df)
    for i in arcid_list:
        temp_df = conc_sampling_point_df[conc_sampling_point_df['ArcID'] == i]
        x_array = temp_df['X-coordinate'].values
        x_array_min = min(x_array)
        x_array_max = max(x_array)
        # 未校正风向
        if i == 1 or i == 2 or i == 3:
            x_add_list = [x_array_min - 2*j for j in range(1, 6)]
        elif i == 4:
            x_add_list = [x_array_min - 2*j for j in range(1, 9)]
        else:
            x_add_list = [x_array_min - j for j in range(1, 13)]

        # 校正风向
        # if i == 1:
        #     x_add_list = [x_array_max + 2 * j for j in range(1, 7)]
        # elif i == 2:
        #     x_add_list = [x_array_max + 2 * j for j in range(1, 9)]
        # elif i == 3:
        #     x_add_list = [x_array_max + 2 * j for j in range(1, 11)]
        # elif i == 4:
        #     x_add_list = [x_array_min + 2 * j for j in range(1, 13)]
        # else:
        #     x_add_list = [x_array_min + j for j in range(1, 16)]

        for x_coor in x_add_list:
            one_data = all_sampling_point_df[
                (all_sampling_point_df['X-coordinate'] == x_coor) & (all_sampling_point_df['ArcID'] == i)]
            # print(one_data)

            target_sample_point_df.loc[num] = one_data.values[0]
            num = num + 1
            # target_conc_sample_point_df.append(one_data, ignore_index=False)
    target_sample_point_df['ArcID'] = target_sample_point_df['ArcID'].astype(int)
    target_sample_point_df['X-coordinate'] = target_sample_point_df['X-coordinate'].astype(int)
    target_sample_point_df['Y-coordinate'] = target_sample_point_df['Y-coordinate'].astype(int)
    print(target_sample_point_df)
    # target_sample_point_df.to_csv('test0104gai.csv', encoding='utf-8-sig')
    # print(conc_sampling_point_df)
    # print(all_sampling_point_df)

    # from ResultAnalyze import plotSamplerDistribution
    # plotSamplerDistribution(sampling_point_df, fig_path=True)
    # sampling_point_df.to_csv('exprid'+str(expr_id)+'_samplinginf.csv')
    # print(len(sampling_point_df))
    # print(sampling_point_df[['x_wc', 'y_wc', 'x_lgc', 'y_lgc']])
    start = time.time()
    main(emiss_df, target_sample_point_df, mete_list, N=N, start_time=start_time, dt_min=dt_min,
         mode_type='forward', scenario_type=Scenario,
         traj_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_traj_N' + str(N) + '_' + date + '.csv',
         conc_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_conc_' + date + '.txt',
         count_file_path=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(N) + '_count_' + date + '.txt',
         conc_file_path_b1=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb1_' + date + '.txt',
         conc_file_path_b2=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb2_' + date + '.txt',
         conc_file_path_b3=Scenario + '_' + 'parallel_exprid' + str(expr_id) + '_N' + str(
             N) + '_concb3_' + date + '.txt',
         process_nums=4)
    end = time.time()
    print('total run time:', end - start)
    exit()
    # '''
    '''--------------第17组实验(NCBL)---------------'''




