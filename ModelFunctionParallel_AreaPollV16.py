#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/02/15
# @Author  :
# @Site    :
# @File    : ModelFunctionParallel_AreaPoll20230623.py
# @Software: PyCharm

import os
import math
import numpy as np
import pandas as pd
import json
import datetime
import time
# import csv
# from CalculateStatistics20230215 import CalculateBasicIndex
from CalculateStatistics20230215_SD import CalculateBasicIndex
# from UpdateStatistics20230215 import CalculateInCBL, CalculateInNCBL
# from UpdateStatistics20230215v2 import CalculateInCBL, CalculateInNCBL
# from UpdateStatistics20230215v3 import CalculateInCBL, CalculateInNCBL
# from UpdateStatistics20230215v4 import CalculateInCBL, CalculateInNCBL
# from UpdateStatistics20230215v4_SD import CalculateInCBL, CalculateInNCBL
from UpdateStatistics20230215v4_SDv2 import CalculateInCBL, CalculateInNCBL  # 为解决CBL中前后向模型对等问题
# from UpdateStatistics20230215v4_wb import CalculateInCBL, CalculateInNCBL
# from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
import pathos.multiprocessing as mp
from p_tqdm import p_map

'''忽略粒子y方向的运动（计算过程中每一步都强迫y=0）'''
'''不保存粒子运动路径，减少占用空间'''


'''粒子坐标转换lgc-->wc'''
def transferCoord(x_lgc, y_lgc, alpha):
    # x_wc = x_lgc*math.cos(270-alpha)+y_lgc*math.sin(270-alpha)
    # y_wc = -x_lgc*math.sin(270-alpha)+y_lgc*math.cos(270-alpha)
    x_wc = x_lgc * math.cos(math.radians(270 - alpha)) + y_lgc * math.sin(math.radians(270 - alpha))
    y_wc = -x_lgc * math.sin(math.radians(270 - alpha)) + y_lgc * math.cos(math.radians(270 - alpha))
    return x_wc, y_wc

'''粒子坐标转化wc-->lgc'''
def revTransferCoord(x_wc, y_wc, alpha):
    y_lgc = x_wc * math.sin(math.radians(270 - alpha)) + y_wc * math.cos(math.radians(270 - alpha))
    x_lgc = x_wc * math.cos(math.radians(270 - alpha)) - y_wc * math.sin(math.radians(270 - alpha))
    return x_lgc, y_lgc

'''后向是粒子在所有排放源的上风向(这个判断是在极坐标系下进行，故函数与前向定义域判断一致)'''
'''emission_source_df的坐标已经转换为wc'''
# def whetherInTargetArea(particle_coor, alpha):
#     x_wc, y_wc = particle_coor
#     # print('x_wc:', x_wc, 'y_wc:', y_wc)
#     x_lgc, y_lgc = revTransferCoord(x_wc, y_wc, alpha)
#     # if (x_lgc>=-31) and (x_lgc<=1) and (y_lgc>=-6) and (y_lgc<=6):
#     if (x_lgc >= -31) and (x_lgc <= 1):
#         return True, x_lgc, y_lgc
#     else:
#         return False, x_lgc, y_lgc

# def whetherInTargetArea(particle_coor, alpha):
#     x_wc, y_wc = particle_coor
#     # print('x_wc:', x_wc, 'y_wc:', y_wc)
#     x_lgc, y_lgc = revTransferCoord(x_wc, y_wc, alpha)
#     # if (x_lgc>=-31) and (x_lgc<=1) and (y_lgc>=-6) and (y_lgc<=6):
#     if (x_lgc >= -31) and (x_lgc <= 51):
#         return True, x_lgc, y_lgc
#     else:
#         return False, x_lgc, y_lgc


'''针对（100，0，z）的采样点'''
def whetherInTargetArea(particle_coor, alpha):
    x_wc, y_wc = particle_coor
    # print('x_wc:', x_wc, 'y_wc:', y_wc)
    x_lgc, y_lgc = revTransferCoord(x_wc, y_wc, alpha)
    # if (x_lgc>=-31) and (x_lgc<=1) and (y_lgc>=-6) and (y_lgc<=6):
    # if (x_lgc >= -31) and (x_lgc <= 101):
    # if (x_lgc >= -31) and (x_lgc <= 51):
    if ((x_lgc >= -31) and (x_lgc <= 51)) and ((y_lgc >= -6) and (y_lgc <= 6)):
        return True, x_lgc, y_lgc
    else:
        return False, x_lgc, y_lgc


def whetherInTargetArea_b(particle_coor, alpha):
    x_wc, y_wc = particle_coor
    # print('x_wc:', x_wc, 'y_wc:', y_wc)
    x_lgc, y_lgc = revTransferCoord(x_wc, y_wc, alpha)
    # if (x_lgc >= -31) and (x_lgc <= 51):
    if ((x_lgc >= -31) and (x_lgc <= 51)) and ((y_lgc >= -6) and (y_lgc <= 6)):
        return True, x_lgc, y_lgc
    else:
        return False, x_lgc, y_lgc


'''判断对流或非对流'''
def determineScenario(Z_i, L, u_star, w_star):
    # 对L进行重新计算 1201
    if L < 0 and w_star > 0:
        L = -math.pow(u_star/w_star, 3) * Z_i / 0.4
    if L>0:
        secn_type = 'NCBL'
        # return secn_type
    elif L<0:
        if -Z_i/L < 1000:
            secn_type = 'NCBL'
        else:
            if w_star > 0:
                secn_type = 'CBL'
            else:
                secn_type = 'NCBL'
        # return secn_type
    else:
        print('wrong Monin-Obukhov length (L=0) input')
        # exit()
        return
    return secn_type

'''对流情形：粒子速度初始化'''
def initParticleV_inCBL(sigma_u, sigma_v, sigma_w, U_aver, V_aver, W_aver, u_star, w_star, z, Z_i):
    Z = z / Z_i
    b1, b2, b3, b4 = 0.0020, 1.2, 0.333, 0.72
    w_square_aver = math.pow(w_star, 2) * math.pow((b1 + b2 * Z * (1 - Z) * (1 - b3 * Z)), 2 / 3)
    w_cube_aver = math.pow(w_star, 3) * b4 * Z * (1 - Z) * (1 - b3 * Z)
    w_B_aver = (math.pow(math.pow(w_cube_aver, 2) + 8 * math.pow(w_square_aver, 3), 1 / 2) - w_cube_aver) / (
                4 * w_square_aver)
    w_A_aver = w_square_aver / (2 * w_B_aver)
    A = w_B_aver / (w_A_aver + w_B_aver)
    sigma_u_s = math.pow(sigma_u, 2)
    r = np.random.uniform(low=0, high=1, size=None)
    if r <= A:
        # w = w_A_aver*np.random.normal(loc=w_A_aver, scale=1, size=None)
        w = np.random.normal(loc=w_A_aver, scale=math.pow(w_A_aver, 1/2)) #0211
    else:
        # w = -w_B_aver*np.random.normal(loc=-w_B_aver, scale=1, size=None)
        w = np.random.normal(loc=-w_B_aver, scale=math.pow(w_B_aver, 1/2)) #0211
    # rho = -math.pow(u_star/sigma_w, 2)
    # 20230309更新
    rho = -math.pow(u_star * math.pow(1 - z / Z_i, 3 / 4) / sigma_w, 2)
    c = math.pow((sigma_u_s-math.pow(rho*sigma_w, 2)), 1/2)
    u = rho*w+c*np.random.normal(loc=0, scale=1, size=None)
    u = sigma_u*np.random.normal(loc=0, scale=1, size=None) # 20240406 溯源
    v = sigma_v*np.random.normal(loc=0, scale=1, size=None)
    U = u+U_aver
    V = v+V_aver
    W = w+W_aver
    # return abs(U), abs(V), abs(W)
    return U, V, W

'''非对流情形：粒子速度初始化'''
def initParticleV_inNCBL(sigma_u, sigma_v, sigma_w, u_star, z, Z_i):
    vw_aver = 0
    uv_aver = 0
    sigma_u_s = math.pow(sigma_u, 2)
    sigma_v_s = math.pow(sigma_v, 2)
    sigma_w_s = math.pow(sigma_w, 2)
    w = sigma_w*np.random.normal(loc=0.0, scale=1, size=None)
    # rho = -math.pow(u_star/sigma_w, 2)
    # 20230309更新
    rho = -math.pow(u_star * math.pow(1 - z / Z_i, 3 / 4) / sigma_w, 2)
    u = rho*w+math.pow(sigma_u_s-math.pow(rho*sigma_w, 2), 1/2)*np.random.normal(loc=0.0, scale=1, size=None)
    rho_u = (sigma_w_s * uv_aver + math.pow(u_star, 2) * vw_aver) / (sigma_u_s * sigma_w_s - math.pow(u_star, 4))
    rho_w = (sigma_u_s * vw_aver + math.pow(u_star, 2) * uv_aver) / (sigma_u_s * sigma_w_s - math.pow(u_star, 4))
    v = rho_u * u + rho_w * w + math.pow((sigma_v_s - math.pow(rho_u * sigma_u, 2) - math.pow(rho_w * sigma_w, 2) + 2 * math.pow(u_star,2) * rho_u * rho_w),1 / 2)*np.random.normal(loc=0.0, scale=1, size=None)
    return u, v, w

'''并行计算每个源每个粒子释放情况'''
'''计算非对流情形下单个粒子的轨迹，将其视为一个子任务'''
#1、part_coor = [x_wc, y_wc, z]
#2、Q_ei, N_ei为该排放源的释放粒子数
def singlePartTraj_ForwardInNCBL(traj_file_path, emission_part_id, Q_ei, N_ei, part_coor, sampling_point_df, meteoro_list, start_time, dt_min):
    # with open(traj_file_path, 'a', encoding='utf-8-sig', newline='') as f:
    np.random.seed()
    # f = open(traj_file_path, 'a', encoding='utf-8-sig', newline='') 0214
    # csvwrite = csv.writer(f)  0214
    emission_id, part_id = emission_part_id
    print('emission_id:', emission_id, 'part_id:', part_id)
    secn_type = 'NCBL'
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    result_list = []
    sample_conc_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])  # 记录该粒子对于各采样点的浓度贡献
    # sample_partcount_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))]) # 记录各采样点上经过的粒子数
    x_wc, y_wc, z = part_coor  # 粒子释放的初始坐标
    k, C0 = 0.4, 3.0
    calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
    L_update = calculateBasicIndex.L  # 若满足L<0,w*>0重新进行计算
    sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
    sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
    epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
    U_z = calculateBasicIndex.calculateWD()  # 20221123
    U_aver = U_z
    V_aver, W_aver = 0, 0
    '''速度初始化20221116, 初始化时未考虑粒子反射'''
    u, v, w = initParticleV_inNCBL(sigma_u, sigma_v, sigma_w, u_star, z, Z_i)
    t_update = 0
    bool_value_update, x_lgc, y_lgc = whetherInTargetArea([x_wc, y_wc], alpha)
    # temp_list = [start_time, t_update, emission_id, part_id, x_wc, y_wc, z,
    #              u, v, w, U_aver, '', '', '', sigma_u, sigma_v, sigma_w, epsilon, secn_type]
    # csvwrite.writerow(temp_list) 0214
    # result_list.append(temp_list)  # 记录初始时刻
    # count_nums = 0  # 20230131
    # while count_nums < 3:  # 20230131
    while bool_value_update:
        # t_update = t_update + dt
        '''更新时刻(粒子位置处)的U，V，W'''
        U_aver_update = U_aver  # 暂时简单化处理1127需确认
        V_aver_update, W_aver_update = V_aver, W_aver
        # 记录前一时刻的坐标
        x_wc_old, y_wc_old, z_old = x_wc, y_wc, z
        calculateInNCBL = CalculateInNCBL(u_star=u_star, w_star=w_star,
                                          x_wc=x_wc, y_wc=y_wc, z=z, Z_i=Z_i, L=L_update,
                                          u=u, v=v, w=w,
                                          U_aver_update=U_aver_update, V_aver_update=V_aver_update,
                                          W_aver_update=W_aver_update,
                                          sigma_u=sigma_u, sigma_v=sigma_v,
                                          sigma_w=sigma_w, dt_min=dt_min, epsilon=epsilon, C0=C0)
        dt = calculateInNCBL.dt
        # print('dt:', dt)
        t_update = t_update + dt
        a_u, a_v, a_w, u, v, w, x_wc, y_wc, z = calculateInNCBL.updateXYZ()
        # y_wc = 0 # 20231013 计算过程中的每一次都强迫y=0
        '''判断粒子是否需要进行反射，不改变v'''
        # print('反射前z, z_0, z_i:', z, z_0, z_i)
        z_r = max(z_0, 0.05)
        if z < z_r:
            z = 2 * z_r - z
            u, w = -u, -w
        if z > Z_i:
            z = 2 * Z_i - z
            u, w = -u, -w
        # print('反射后z, z_0, z_i:', z, z_0, z_i)
        # print(' ')
        '''判断粒子是否需要进行反射'''

        '''遍历采样点，判断粒子是否撞击了到了采样点，计算采样点浓度'''
        for m in range(len(sampling_point_df)):
            one_sampling_point_df = sampling_point_df.iloc[m]
            s_x_wc, s_y_wc, s_z, s_x_lgc = one_sampling_point_df['x_wc'], one_sampling_point_df['y_wc'], \
                                           one_sampling_point_df['z'], one_sampling_point_df['x_lgc']
            # 判断第i个排放源的第j个粒子是否要纳入第m个采样点处的浓度计算,判断条件确认
            # 需要人为设定20221127？？？
            '''
            if s_arc_id == 1 or 2 or 3:
                delta_y, delat_z = 0.2, 0.2
            else:
                delta_y, delat_z = 0.5, 0.5
            '''
            if s_x_lgc == 0:
                delta_y, delta_z = 0.2, 0.2
            elif s_x_lgc == 25:
                delta_y, delta_z = 0.5, 0.3
            else: #  50
                delta_y, delta_z = 1, 0.4
            # if s_x_wc > x_wc_old and s_x_wc <= x_wc:
            if ((s_x_wc > x_wc_old) and (s_x_wc <= x_wc)) or ((s_x_wc < x_wc_old) and (s_x_wc >= x_wc)):
                y_star = y_wc_old + (y_wc - y_wc_old) / (x_wc - x_wc_old) * (s_x_wc - x_wc_old)
                z_star = z_old + (z - z_old) / (x_wc - x_wc_old) * (s_x_wc - x_wc_old)
                '''20231013'''
                '''
                if z_star >= s_z - 1 / 2 * delta_z and z_star <= s_z + 1 / 2 * delta_z: # 去掉与delta_y相关的判断
                    sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei/N_ei)*(1/(abs(U_z+u)*delta_y*delta_z))  # 20240222 重新加回delta_y
                '''
                # '''
                if y_star >= s_y_wc - 1 / 2 * delta_y and y_star <= s_y_wc + 1 / 2 * delta_y:
                    if z_star >= s_z - 1 / 2 * delta_z and z_star <= s_z + 1 / 2 * delta_z:
                        sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei/N_ei) * (1/(abs(U_z)*delta_y*delta_z)) # 20240226
                        # sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei/N_ei) * (1/(abs(U_z+u)*delta_y*delta_z)) # 用粒子总速度替代平均风速, 20240222 重新加回delta_y
                        # sample_partcount_dic[str(m)] = sample_partcount_dic[str(m)] + 1
                # '''
        # csvwrite.writerow('#')
        bool_value_update, x_lgc, y_lgc = whetherInTargetArea([x_wc, y_wc], alpha)
        # print('bool_value_update:', bool_value_update)
        '''更新气象相关参数(目前的数据无需更新)'''
        # k, C0 = 0.4, 3.0
        calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
        sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
        sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
        epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
        U_z = calculateBasicIndex.calculateWD()  # 20221123
        U_aver = U_z
        V_aver, W_aver = 0, 0
        '''保存当前时刻计算得到的x_wc, y_wc, z, u, v, w'''
        # ['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
        # 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario']
        # temp_list = [start_time + pd.Timedelta(t_update, unit='s'), t_update, emission_id, part_id, x_wc, y_wc, z,
        #              u, v, w, U_aver_update, a_u, a_v, a_w, sigma_u, sigma_v, sigma_w, epsilon, secn_type]
        # result_list.append(temp_list)  # 记录更新时刻
        # csvwrite.writerow(temp_list) 0214
        # count_nums = count_nums + 1  # 20230131
    # result_df = pd.DataFrame(np.array(result_list),
    #                          columns=['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z',
    #                                   'u', 'v', 'w', 'U_aver', 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w',
    #                                   'epsilon', 'scenario'])
    # sample_conc_dic 存储单个粒子对于全部采样点的浓度贡献
    # result_df 存储单个粒子的轨迹
    # return sample_conc_dic, sample_partcount_dic, result_df
    return sample_conc_dic

'''并行的计算每个源每个粒子的轨迹'''
def forwardInNCBLParallel(emission_source_df, sampling_point_df, meteoro_list, N, start_time, dt_min, traj_file_path,
                          conc_file_path, count_file_path, process_nums):
    # secn_type = 'NCBL'
    # print('particle_array:', particle_array)
    '''所有并行的粒子轨迹结果均写入下述文件中''' #0214
    # f = open(traj_file_path, 'w', encoding='utf-8-sig', newline='')
    # csvwrite = csv.writer(f)
    # csvwrite.writerow(['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
    #                    'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario'])
    emission_source_Qs = emission_source_df['Qs'].values
    emission_source_xyz = emission_source_df[['x_wc', 'y_wc', 'z']].values
    '''构建emission_part_id_list'''
    emission_part_id_list = [[0, _] for _ in range(len(emission_source_df))] #仅针对面源
    Q_ei_list = list(emission_source_Qs)
    N_ei_list = [N for _ in range(len(emission_source_df))]
    part_coor_list = [list(_) for _ in emission_source_xyz]
    task_nums = len(emission_part_id_list)
    print('task_nums:', task_nums) # 即粒子数

    traj_file_path_list = [traj_file_path for _ in range(task_nums)]
    # emission_part_id_list = [] 存储不同任务
    # Q_ei_list = []
    # N_ei_list = []
    # part_coor_list = []
    sampling_point_df_list = [sampling_point_df for _ in range(task_nums)]
    meteoro_list_list = [meteoro_list for _ in range(task_nums)]
    start_time_list = [start_time for _ in range(task_nums)]
    dt_min_list = [dt_min for _ in range(task_nums)]
    cores = mp.cpu_count()
    print('cores:', cores)
    pool = mp.ProcessingPool(processes=process_nums)
    result_list = p_map(singlePartTraj_ForwardInNCBL, traj_file_path_list, emission_part_id_list,
                        Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list,
                        meteoro_list_list, start_time_list, dt_min_list)
    # pool = Pool(processes=process_nums)
    # result_list = pool.map(singlePartTraj_ForwardInNCBL, traj_file_path_list, emission_part_id_list,
    #                        Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list,
    #                        meteoro_list_list, start_time_list, dt_min_list)
    # result_list = list(pool.imap(singlePartTraj_ForwardInNCBL, traj_file_path_list, emission_part_id_list,
    #                              Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list, meteoro_list_list,
    #                              start_time_list, dt_min_list))

    # args_list = zip(traj_file_path_list, emission_part_id_list,Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list, meteoro_list_list, start_time_list, dt_min_list)
    # result_list = pool.starmap_async(singlePartTraj_ForwardInNCBL, args_list).get()
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程退出
    # f.close()
    # print(result_list)
    dic_list = []
    dic_value_list = []
    partcount_dic_list = []
    partcount_dic_value_list = []
    df_list = []

    '''6月7日改进数据处理，利用生成式列表'''
    # 待进行测试
    '''
    df_list = [item[2] for item in result_list]  # 轨迹
    dic_list = [item[0] for item in result_list]
    dic_value_list = [list((item[0]).values()) for item in result_list]  # 浓度
    partcount_dic_list = [item[1] for item in result_list]
    partcount_dic_value_list = [list((item[1]).values()) for item in result_list]  # 粒子数
    '''
    # '''
    for i in range(task_nums):
        dic = result_list[i]
        # sample_partcount_dic = result_list[i][1]
        # df = result_list[i][2]
        dic_list.append(dic)
        dic_value_list.append(list(dic.values()))
        # partcount_dic_list.append(sample_partcount_dic)
        # partcount_dic_value_list.append(list(sample_partcount_dic.values()))
        # df_list.append(df)
    # '''
    dic_total_value_list = list(np.array(dic_value_list).sum(axis=0))
    # partcount_dic_total_value_list = list(np.array(partcount_dic_value_list).sum(axis=0))
    # print(dic_total_value_list)
    # print(len(dic_total_value_list))
    # exit()
    '''合并浓度, 合并粒子轨迹结果'''
    print('--------------------------')
    dic_total = dict([(str(_), float(dic_total_value_list[_])) for _ in range(len(dic_total_value_list))])  # 记录该粒子对于各采样点的浓度贡献
    js = json.dumps(dic_total)
    f = open(conc_file_path, 'w')
    f.write(js)
    f.close()

    # partcount_dic_total = dict(
    #     [(str(_), float(partcount_dic_total_value_list[_])) for _ in range(len(partcount_dic_total_value_list))])
    # js = json.dumps(partcount_dic_total)
    # f = open(count_file_path, 'w')
    # f.write(js)
    # f.close()

    # df_total = pd.concat(df_list, axis=0)
    # df_total.to_csv('v2_'+traj_file_path, encoding='utf-8-sig') #0214
    # df_total.to_csv(traj_file_path, encoding='utf-8-sig') #0214

    return

'''并行计算每个源每个粒子释放情况'''
'''计算对流情形下单个粒子的轨迹，将其视为一个子任务'''
#1、part_coor = [x_wc, y_wc, z]
#2、Q_ei, N_ei为该排放源的释放粒子数
def singlePartTraj_ForwardInCBL(traj_file_path, emission_part_id, Q_ei, N_ei, part_coor, sampling_point_df, meteoro_list, start_time, dt_min):
    # with open(traj_file_path, 'a', encoding='utf-8-sig', newline='') as f:
    np.random.seed()
    # f = open(traj_file_path, 'a', encoding='utf-8-sig', newline='') # 0214
    # csvwrite = csv.writer(f) # 0214
    emission_id, part_id = emission_part_id
    print('emission_id:', emission_id, 'part_id:', part_id)
    secn_type = 'CBL'
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    result_list = []
    sample_conc_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])  # 记录该粒子对于各采样点的浓度贡献
    # sample_partcount_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])  # 记录各采样点上经过的粒子数
    x_wc, y_wc, z = part_coor  # 粒子释放的初始坐标
    k, C0 = 0.4, 3.0
    calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
    L = calculateBasicIndex.L  # 若满足L<0,w*>0重新进行计算
    sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
    sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
    pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
    epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
    U_z = calculateBasicIndex.calculateWD()  # 20221123
    U_aver = U_z
    V_aver, W_aver = 0, 0
    '''速度初始化20221116, 初始化时未考虑粒子反射'''
    U, V, W = initParticleV_inCBL(sigma_u, sigma_v, sigma_w, U_aver, V_aver, W_aver, u_star, w_star, z, Z_i)
    '''速度初始化20221116, 初始化时未考虑粒子反射'''

    t_update = 0
    bool_value_update, x_lgc, y_lgc = whetherInTargetArea([x_wc, y_wc], alpha)
    # temp_list = [start_time, t_update, emission_id, part_id, x_wc, y_wc, z,
    #              U, V, W, U_aver, '', '', '', sigma_u, sigma_v, sigma_w,
    #              '', '', '', '', '', '', '', '', '', '', '',
    #              '', '', epsilon, secn_type]
    # csvwrite.writerow(temp_list) 0214
    # result_list.append(temp_list)  # 记录初始时刻
    while bool_value_update:
        '''原位置处的更新时刻的U，V，W'''
        U_aver_update = U_aver  # 暂时简单化处理1127需确认
        V_aver_update, W_aver_update = V_aver, W_aver
        # 记录前一时刻的坐标
        x_wc_old, y_wc_old, z_old = x_wc, y_wc, z
        '''根据应用场景计算x,y,z,u,v,w'''
        calculateInCBL = CalculateInCBL(u_star=u_star, w_star=w_star, x_wc=x_wc, y_wc=y_wc, z=z,
                                        Z_i=Z_i, U=U, V=V, W=W,
                                        U_aver=U_aver_update, V_aver=V_aver_update, W_aver=W_aver_update,
                                        sigma_u=sigma_u, sigma_v=sigma_v, sigma_w=sigma_w,
                                        epsilon=epsilon, pd_U_aver_to_z=pd_U_aver_to_z,
                                        dt_min=dt_min, C0=C0)
        dt = calculateInCBL.dt
        t_update = t_update + dt
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, x_wc, y_wc, z = calculateInCBL.updateXYZ()
        # y_wc = 0  # 20231013 计算过程中的每一次都强迫y=0
        '''判断粒子是否需要进行反射，不改变v'''
        # print('反射前z, z_0, z_i:', z, z_0, z_i)
        z_r = max(z_0, 0.05)
        if z < z_r:
            z = 2 * z_r - z
            U, W = -U+2*U_aver, -W+2*W_aver
        if z > Z_i:
            z = 2 * Z_i - z
            U, W = -U+2*U_aver, -W+2*W_aver
        # print('反射后z, z_0, z_i:', z, z_0, z_i)
        # print(' ')
        '''判断粒子是否需要进行反射'''

        '''遍历采样点，判断粒子是否撞击了到了采样点，计算采样点浓度'''
        '''11月9日需要进行确认'''
        for m in range(len(sampling_point_df)):
            one_sampling_point_df = sampling_point_df.iloc[m]
            s_x_wc, s_y_wc, s_z, s_x_lgc = one_sampling_point_df['x_wc'], one_sampling_point_df['y_wc'], \
                                           one_sampling_point_df['z'], one_sampling_point_df['x_lgc']
            # 判断第i个排放源的第j个粒子是否要纳入第m个采样点处的浓度计算,判断条件确认
            # 需要人为设定20221127？？？
            # delta_y, delta_z = 0.2, 0.2
            # delta_y, delta_z = 1, 0.4
            if s_x_lgc == 0:
                delta_y, delta_z = 0.2, 0.2
            elif s_x_lgc == 25:
                delta_y, delta_z = 0.5, 0.3
            else: #  50
                delta_y, delta_z = 1, 0.4
            # print(delta_y, delta_z)
            # if s_x_wc > x_wc_old and s_x_wc <= x_wc:
            if ((s_x_wc > x_wc_old) and (s_x_wc <= x_wc)) or ((s_x_wc < x_wc_old) and (s_x_wc >= x_wc)):
                y_star = y_wc_old + (y_wc - y_wc_old) / (x_wc - x_wc_old) * (s_x_wc - x_wc_old)
                z_star = z_old + (z - z_old) / (x_wc - x_wc_old) * (s_x_wc - x_wc_old)
                '''20231013 修改'''
                '''
                if z_star >= s_z - 1 / 2 * delta_z and z_star <= s_z + 1 / 2 * delta_z:
                    sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei / N_ei) * (1 / (abs(U) * delta_z))  # 用粒子总速度替代平均风速
                '''
                # '''
                # 添加回delta_y, 取消y_wc=0 20240301？
                if y_star >= s_y_wc - 1 / 2 * delta_y and y_star <= s_y_wc + 1 / 2 * delta_y:
                    if z_star >= s_z - 1 / 2 * delta_z and z_star <= s_z + 1 / 2 * delta_z:
                        # sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei/N_ei) * (1/(abs(U_z)*delta_y*delta_z))
                        sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei/N_ei) * (1/(abs(U)*delta_y*delta_z))  # 用粒子总速度替代平均风速，20240306
                        # sample_partcount_dic[str(m)] = sample_partcount_dic[str(m)]+1
                # '''
        # csvwrite.writerow('#')
        bool_value_update,x_lgc,y_lgc = whetherInTargetArea([x_wc, y_wc], alpha)
        # print('bool_value_update:', bool_value_update)
        '''更新气象相关参数(目前的数据无需更新)'''
        # k, C0 = 0.4, 3
        calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
        sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
        sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
        pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
        epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
        U_z = calculateBasicIndex.calculateWD()  # 20221123
        U_aver = U_z
        V_aver, W_aver = 0, 0
        '''保存当前时刻计算得到的x_wc, y_wc, z, u, v, w'''
        # ['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
        # 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario']
        # temp_list = [start_time + pd.Timedelta(t_update, unit='s'), t_update, emission_id, part_id, x_wc, y_wc, z,
        #              U, V, W, U_aver_update, a_U, a_V, a_W, sigma_u, sigma_v, sigma_w,
        #              g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1,
        #              P_B_part2, w_A_aver, w_B_aver,
        #              epsilon, secn_type]
        # result_list.append(temp_list)  # 记录更新时刻
        # csvwrite.writerow(temp_list) #0214
        # count_nums = count_nums+1  # 20230131
    # result_df = pd.DataFrame(np.array(result_list),
    #                          columns=['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'U', 'V', 'W', 'U_aver',
    #                                   'a_U', 'a_V', 'a_W', 'sigma_u', 'sigma_v', 'sigma_w',
    #                                   'g_w', 'g_u', 'g_a', 'A', 'B', 'P_A', 'P_A_part1', 'P_A_part2', 'P_B', 'P_B_part1', 'P_B_part2',
    #                                   'w_A_aver/sigma_A', 'w_B_aver/sigma_B',
    #                                   'epsilon', 'scenario'])
    # sample_conc_dic 存储单个粒子对于全部采样点的浓度贡献
    # result_df 存储单个粒子的轨迹
    # return sample_conc_dic, sample_partcount_dic, result_df
    return sample_conc_dic

'''并行计算每个源每个粒子释放情况'''
'''计算对流情形下单个粒子的轨迹，将其视为一个子任务'''
#1、part_coor = [x_wc, y_wc, z]
#2、Q_ei, N_ei为该排放源的释放粒子数
def singlePartTraj_ForwardInCBLv2(traj_file_path, emission_part_id, Q_ei, N_ei, part_coor, sampling_point_df, meteoro_list, start_time, dt_min):
    # with open(traj_file_path, 'a', encoding='utf-8-sig', newline='') as f:
    np.random.seed()
    # f = open(traj_file_path, 'a', encoding='utf-8-sig', newline='') # 0214
    # csvwrite = csv.writer(f) # 0214
    emission_id, part_id = emission_part_id
    print('emission_id:', emission_id, 'part_id:', part_id)
    secn_type = 'CBL'
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    result_list = []
    sample_conc_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])  # 记录该粒子对于各采样点的浓度贡献
    # sample_partcount_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])  # 记录各采样点上经过的粒子数
    x_wc, y_wc, z = part_coor  # 粒子释放的初始坐标
    k, C0 = 0.4, 3.0
    calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
    L = calculateBasicIndex.L  # 若满足L<0,w*>0重新进行计算
    sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
    sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
    pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
    epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
    U_z = calculateBasicIndex.calculateWD()  # 20221123
    U_aver = U_z
    V_aver, W_aver = 0, 0
    '''速度初始化20221116, 初始化时未考虑粒子反射'''
    U, V, W = initParticleV_inCBL(sigma_u, sigma_v, sigma_w, U_aver, V_aver, W_aver, u_star, w_star, z, Z_i)
    '''速度初始化20221116, 初始化时未考虑粒子反射'''

    t_update = 0
    bool_value_update, x_lgc, y_lgc = whetherInTargetArea([x_wc, y_wc], alpha)
    # temp_list = [start_time, t_update, emission_id, part_id, x_wc, y_wc, z,
    #              U, V, W, U_aver, '', '', '', sigma_u, sigma_v, sigma_w,
    #              '', '', '', '', '', '', '', '', '', '', '',
    #              '', '', epsilon, secn_type]
    # csvwrite.writerow(temp_list) 0214
    # result_list.append(temp_list)  # 记录初始时刻
    while bool_value_update:
        try:
            '''原位置处的更新时刻的U，V，W'''
            U_aver_update = U_aver  # 暂时简单化处理1127需确认
            V_aver_update, W_aver_update = V_aver, W_aver
            # 记录前一时刻的坐标
            x_wc_old, y_wc_old, z_old = x_wc, y_wc, z
            '''根据应用场景计算x,y,z,u,v,w'''
            calculateInCBL = CalculateInCBL(u_star=u_star, w_star=w_star, x_wc=x_wc, y_wc=y_wc, z=z,
                                            Z_i=Z_i, U=U, V=V, W=W,
                                            U_aver=U_aver_update, V_aver=V_aver_update, W_aver=W_aver_update,
                                            sigma_u=sigma_u, sigma_v=sigma_v, sigma_w=sigma_w,
                                            epsilon=epsilon, pd_U_aver_to_z=pd_U_aver_to_z,
                                            dt_min=dt_min, C0=C0)
            dt = calculateInCBL.dt
            t_update = t_update + dt
            g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, x_wc, y_wc, z = calculateInCBL.updateXYZ()
            # y_wc = 0  # 20231013 计算过程中的每一次都强迫y=0
            '''判断粒子是否需要进行反射，不改变v'''
            # print('反射前z, z_0, z_i:', z, z_0, z_i)
            z_r = max(z_0, 0.05)
            if z < z_r:
                z = 2 * z_r - z
                U, W = -U+2*U_aver, -W+2*W_aver
            if z > Z_i:
                z = 2 * Z_i - z
                U, W = -U+2*U_aver, -W+2*W_aver
            # print('反射后z, z_0, z_i:', z, z_0, z_i)
            # print(' ')
            '''判断粒子是否需要进行反射'''

            '''遍历采样点，判断粒子是否撞击了到了采样点，计算采样点浓度'''
            '''11月9日需要进行确认'''
            for m in range(len(sampling_point_df)):
                one_sampling_point_df = sampling_point_df.iloc[m]
                s_x_wc, s_y_wc, s_z, s_x_lgc = one_sampling_point_df['x_wc'], one_sampling_point_df['y_wc'], \
                                               one_sampling_point_df['z'], one_sampling_point_df['x_lgc']
                # 判断第i个排放源的第j个粒子是否要纳入第m个采样点处的浓度计算,判断条件确认
                # 需要人为设定20221127？？？
                # delta_y, delta_z = 0.2, 0.2
                # delta_y, delta_z = 1, 0.4
                if s_x_lgc == 0:
                    delta_y, delta_z = 0.2, 0.2
                elif s_x_lgc == 25:
                    delta_y, delta_z = 0.5, 0.3
                else: #  50
                    delta_y, delta_z = 1, 0.4
                # print(delta_y, delta_z)
                # if s_x_wc > x_wc_old and s_x_wc <= x_wc:
                if ((s_x_wc > x_wc_old) and (s_x_wc <= x_wc)) or ((s_x_wc < x_wc_old) and (s_x_wc >= x_wc)):
                    y_star = y_wc_old + (y_wc - y_wc_old) / (x_wc - x_wc_old) * (s_x_wc - x_wc_old)
                    z_star = z_old + (z - z_old) / (x_wc - x_wc_old) * (s_x_wc - x_wc_old)
                    '''20231013 修改'''
                    '''
                    if z_star >= s_z - 1 / 2 * delta_z and z_star <= s_z + 1 / 2 * delta_z:
                        sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei / N_ei) * (1 / (abs(U) * delta_z))  # 用粒子总速度替代平均风速
                    '''
                    # '''
                    # 添加回delta_y, 取消y_wc=0 20240301？
                    if y_star >= s_y_wc - 1 / 2 * delta_y and y_star <= s_y_wc + 1 / 2 * delta_y:
                        if z_star >= s_z - 1 / 2 * delta_z and z_star <= s_z + 1 / 2 * delta_z:
                            # sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei/N_ei) * (1/(abs(U_z)*delta_y*delta_z))
                            sample_conc_dic[str(m)] = sample_conc_dic[str(m)] + (Q_ei/N_ei) * (1/(abs(U)*delta_y*delta_z))  # 用粒子总速度替代平均风速，20240306
                            # sample_partcount_dic[str(m)] = sample_partcount_dic[str(m)]+1
                    # '''
            # csvwrite.writerow('#')
            bool_value_update,x_lgc,y_lgc = whetherInTargetArea([x_wc, y_wc], alpha)
            # print('bool_value_update:', bool_value_update)
            '''更新气象相关参数(目前的数据无需更新)'''
            # k, C0 = 0.4, 3
            calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
            sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
            sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
            pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
            epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
            U_z = calculateBasicIndex.calculateWD()  # 20221123
            U_aver = U_z
            V_aver, W_aver = 0, 0
            '''保存当前时刻计算得到的x_wc, y_wc, z, u, v, w'''
            # ['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
            # 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario']
            # temp_list = [start_time + pd.Timedelta(t_update, unit='s'), t_update, emission_id, part_id, x_wc, y_wc, z,
            #              U, V, W, U_aver_update, a_U, a_V, a_W, sigma_u, sigma_v, sigma_w,
            #              g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1,
            #              P_B_part2, w_A_aver, w_B_aver,
            #              epsilon, secn_type]
            # result_list.append(temp_list)  # 记录更新时刻
            # csvwrite.writerow(temp_list) #0214
            # count_nums = count_nums+1  # 20230131
        except Exception as e:
            print(e)
            break
    # result_df = pd.DataFrame(np.array(result_list),
    #                          columns=['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'U', 'V', 'W', 'U_aver',
    #                                   'a_U', 'a_V', 'a_W', 'sigma_u', 'sigma_v', 'sigma_w',
    #                                   'g_w', 'g_u', 'g_a', 'A', 'B', 'P_A', 'P_A_part1', 'P_A_part2', 'P_B', 'P_B_part1', 'P_B_part2',
    #                                   'w_A_aver/sigma_A', 'w_B_aver/sigma_B',
    #                                   'epsilon', 'scenario'])
    # sample_conc_dic 存储单个粒子对于全部采样点的浓度贡献
    # result_df 存储单个粒子的轨迹
    # return sample_conc_dic, sample_partcount_dic, result_df
    return sample_conc_dic


'''并行的计算每个源每个粒子的轨迹'''
def forwardInCBLParallel(emission_source_df, sampling_point_df, meteoro_list, N, start_time, dt_min, traj_file_path,
                         conc_file_path, count_file_path, process_nums):
    # secn_type = 'NCBL'
    # print('particle_array:', particle_array)
    '''所有并行的粒子轨迹结果均写入下述文件中''' #0214
    # f = open(traj_file_path, 'w', encoding='utf-8-sig', newline='')
    # csvwrite = csv.writer(f)
    # csvwrite.writerow(['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'U', 'V', 'W', 'U_aver',
    #                    'a_U', 'a_V', 'a_W', 'sigma_u', 'sigma_v', 'sigma_w',
    #                    'g_w', 'g_u', 'g_a', 'A', 'B', 'P_A', 'P_A_part1', 'P_A_part2', 'P_B', 'P_B_part1', 'P_B_part2',
    #                    'w_A_aver/sigma_A', 'w_B_aver/sigma_B',
    #                    'epsilon', 'scenario'])

    emission_source_Qs = emission_source_df['Qs'].values
    emission_source_xyz = emission_source_df[['x_wc', 'y_wc', 'z']].values
    emission_part_id_list = [[0, _] for _ in range(len(emission_source_df))]  # 仅针对面源
    Q_ei_list = list(emission_source_Qs)
    N_ei_list = [N for _ in range(len(emission_source_df))]
    part_coor_list = [list(_) for _ in emission_source_xyz]
    task_nums = len(emission_part_id_list)
    print('task_nums:', task_nums)

    traj_file_path_list = [traj_file_path for _ in range(task_nums)]
    # emission_part_id_list = [] 存储不同任务
    # Q_ei_list = []
    # N_ei_list = []
    # part_coor_list = []
    sampling_point_df_list = [sampling_point_df for _ in range(task_nums)]
    meteoro_list_list = [meteoro_list for _ in range(task_nums)]
    start_time_list = [start_time for _ in range(task_nums)]
    dt_min_list = [dt_min for _ in range(task_nums)]
    cores = mp.cpu_count()
    print('cores:', cores)
    pool = mp.ProcessingPool(processes=process_nums)

    # result_list = p_map(singlePartTraj_ForwardInCBLv2, traj_file_path_list, emission_part_id_list,
    #                     Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list,
    #                     meteoro_list_list, start_time_list, dt_min_list)
    # print('len(result_list):', len(result_list))  # 返回任务数，即正常运行的粒子数
    result_list = p_map(singlePartTraj_ForwardInCBL, traj_file_path_list, emission_part_id_list,
                        Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list,
                        meteoro_list_list, start_time_list, dt_min_list)
    # args_zip = zip(traj_file_path_list, emission_part_id_list, Q_ei_list, N_ei_list, part_coor_list,
    #                sampling_point_df_list, meteoro_list_list, start_time_list, dt_min_list)
    # result_list = pool.starmap_async(singlePartTraj_ForwardInCBL, args_zip).get()
    # f.close()
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()   # 主进程阻塞等待子进程退出
    # sample_conc_dic, sample_partcount_dic, result_df
    # print(result_list)
    dic_list = []
    dic_value_list = []
    partcount_dic_list = []
    partcount_dic_value_list = []
    df_list = []

    '''6月7日改进数据处理，利用生成式列表'''
    # 待进行测试
    # df_list = [item[2] for item in result_list]  # 轨迹
    dic_list = result_list
    dic_value_list = [list(item.values()) for item in result_list]  # 浓度
    # partcount_dic_list = [item[1] for item in result_list]
    # partcount_dic_value_list = [list((item[1]).values()) for item in result_list]  # 粒子数
    '''
    for i in range(task_nums):
        dic = result_list[i][0]
        sample_partcount_dic = result_list[i][1]
        df = result_list[i][2]
        # dic_list.append(dic)
        dic_value_list.append(list(dic.values()))
        partcount_dic_list.append(sample_partcount_dic)
        partcount_dic_value_list.append(list(sample_partcount_dic.values()))
        df_list.append(df)
    '''
    dic_total_value_list = list(np.array(dic_value_list).sum(axis=0))
    # partcount_dic_total_value_list = list(np.array(partcount_dic_value_list).sum(axis=0))
    # print(dic_total_value_list)
    # print(len(dic_total_value_list))
    # exit()
    '''合并浓度, 合并粒子轨迹结果'''
    print('--------------------------')
    dic_total = dict([(str(_), float(dic_total_value_list[_])) for _ in range(len(dic_total_value_list))])  # 记录该粒子对于各采样点的浓度贡献
    js = json.dumps(dic_total)
    f = open(conc_file_path, 'w')
    f.write(js)
    f.close()
    # partcount_dic_total = dict([(str(_), float(partcount_dic_total_value_list[_])) for _ in range(len(partcount_dic_total_value_list))])
    # js = json.dumps(partcount_dic_total)
    # f = open(count_file_path, 'w')
    # f.write(js)
    # f.close()
    # df_total = pd.concat(df_list, axis=0)
    # # df_total.to_csv('v2_'+traj_file_path, encoding='utf-8-sig') # 0214
    # df_total.to_csv(traj_file_path, encoding='utf-8-sig', index=False)

    return




'''并行计算每个源每个粒子释放情况'''
'''计算非对流情形下单个粒子的轨迹，将其视为一个子任务'''
def singlePartTraj_BackwardInNCBL(traj_file_path, emiss_info_list, sampler_part_id, N_si, part_coor_b, sampling_point_df, meteoro_list, start_time, dt_min):
    np.random.seed()
    # f = open(traj_file_path, 'a', encoding='utf-8-sig', newline='') 0214
    # csvwrite = csv.writer(f)  0214
    mu_w, z_s, Q_s = emiss_info_list
    sampler_id, part_id = sampler_part_id
    print('sampler_id:', sampler_id, 'part_id:', part_id)
    secn_type = 'NCBL'
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    result_list = []
    # 记录回溯粒子（即从某排放源释放的粒子）对于各采样点的浓度贡献
    sample_conc_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])
    # 用于计算浓度方式2、方式3
    sample_w_dic = dict([(str(_), []) for _ in range(len(sampling_point_df))])
    # 记录各采样点i回溯到各源j处的粒子数
    # sample_partcount_dic = dict([(str(i)+'-'+str(j), 0) for i in range(len(sampling_point_df)) for j in range(len(emiss_df))])
    x_wc, y_wc, z = part_coor_b  # 粒子释放的初始坐标及采样点位置
    k, C0 = 0.4, 3.0
    calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
    L_update = calculateBasicIndex.L  # 若满足L<0,w*>0重新进行计算
    sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
    sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
    epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
    U_z = calculateBasicIndex.calculateWD()  # 20221123
    U_aver = U_z
    V_aver, W_aver = 0, 0
    '''速度初始化20221116, 初始化时未考虑粒子反射'''
    u, v, w = initParticleV_inNCBL(sigma_u, sigma_v, sigma_w, u_star, z, Z_i)
    t_update = 0
    bool_value_update, x_lgc, y_lgc = whetherInTargetArea_b([x_wc, y_wc], alpha) #后向
    temp_list = [start_time, t_update, sampler_id, part_id, x_wc, y_wc, z,
                 u, v, w, U_aver, '', '', '', sigma_u, sigma_v, sigma_w, epsilon, secn_type]
    # csvwrite.writerow(temp_list) 0214
    result_list.append(temp_list)  # 记录初始时刻
    another_bool_value_update = True

    while (bool_value_update & another_bool_value_update):  # 有一个不满足则停止
        # t_update = t_update + dt
        '''原位置处的更新时刻的U，V，W'''
        U_aver_update = U_aver  # 暂时简单化处理1127需确认
        V_aver_update, W_aver_update = V_aver, W_aver
        # 记录前一时刻的坐标
        x_wc_old, y_wc_old, z_old = x_wc, y_wc, z  # 在后向，若x_wc视为t时刻，则x_wc_old视为t+1时刻，
        calculateInNCBL = CalculateInNCBL(u_star=u_star, w_star=w_star,
                                          x_wc=x_wc, y_wc=y_wc, z=z, Z_i=Z_i, L=L_update,
                                          u=u, v=v, w=w,
                                          U_aver_update=U_aver_update, V_aver_update=V_aver_update,
                                          W_aver_update=W_aver_update,
                                          sigma_u=sigma_u, sigma_v=sigma_v,
                                          sigma_w=sigma_w, dt_min=dt_min, epsilon=epsilon, C0=C0)
        dt = calculateInNCBL.dt
        # print('dt:', dt)
        t_update = t_update + dt
        a_u, a_v, a_w, u, v, w, x_wc, y_wc, z = calculateInNCBL.updateXYZ_b() # 后向
        # y_wc = 0 # 20231013
        '''遍历采样点，判断粒子是否撞击了到了污染源，若是则计算从该采样点释放的粒子（等价于从污染源释放）对于计算采样点浓度'''
        # print('sampling_point_df.iloc[sampler_id]:', sampling_point_df.iloc[sampler_id])
        e_A = 300 # 30*10，源面积
        # e_A = 30 # 20231117
        e_Rs = math.pow(e_A/math.pi, 1/2)
        e_mu_w = mu_w # 提前计算好排放源处的sigma_w
        e_z = z_s
        e_Q = Q_s
        # 在后向，带old为更新前，无old为更新后
        '''20231013'''
        '''
        if (x_lgc>=-30) and (x_lgc<=0): # 去掉与y相关的判断
            if (z <= e_z) and (z_old > e_z):
                sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1 / N_si * e_Q / e_A * math.pow(2 * math.pi, 1 / 2) / e_mu_w
                sample_w_dic[str(sampler_id)].append(w)
        '''
        # '''
        if (x_lgc>=-30) and (x_lgc<=0) and (y_lgc>=-5) and (y_lgc<=5):
            if (z <= e_z) and (z_old > e_z):
                sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1 / N_si * e_Q / e_A * math.pow(2 * math.pi, 1 / 2) / e_mu_w
                sample_w_dic[str(sampler_id)].append(w)
        # '''
        if x_lgc <= -31:
            another_bool_value_update = False  # 判断是否离开

        '''
        if (z > e_z) and (z_old <= e_z) and (x_lgc>=-30) and (x_lgc<=0) and (y_lgc>=-5) and (y_lgc<=5):
            X = x_wc+(x_wc_old-x_wc)/(z_old-z)*(e_z-z)
            Y = y_wc+(y_wc_old-y_wc)/(z_old-z)*(e_z-z)
            # 不允许粒子反复贡献
            if X >= -e_Rs:
                sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)]+1/N_si*e_Q/e_A*math.pow(2*math.pi, 1/2)/e_mu_w
                # sample_partcount_dic[str(sampler_id)+'-'+str(m)] = sample_partcount_dic[str(sampler_id)+'-'+str(m)]+1
                sample_w_dic[str(sampler_id)].append(w)
            if X < -e_Rs:
                another_bool_value_update = False  # 判断是否离开
        '''
            #允许粒子反复贡献
            # if determine_value <= e_Rs:
            #     sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)]+1/N_si*e_Q/e_A*math.pow(2*math.pi, 1/2)/e_mu_w
            #     sample_partcount_dic[str(sampler_id)+'-'+str(m)] = sample_partcount_dic[str(sampler_id)+'-'+str(m)]+1
            #
                # sample_partcount_dic[str(m)] = sample_partcount_dic[str(m)] + 1

        '''判断粒子是否需要进行反射，不改变v'''
        # print('反射前z:', z)
        z_r = max(z_0, 0.05)
        if z < z_r:
            z = 2 * z_r - z
            u, w = -u, -w
        if z > Z_i:
            z = 2 * Z_i - z
            u, w = -u, -w
        # print('反射后z, z_r, Z_i:', z, z_r, Z_i)
        # print(' ')
        '''判断粒子是否需要进行反射'''

        # csvwrite.writerow('#')
        bool_value_update, x_lgc, y_lgc = whetherInTargetArea_b([x_wc, y_wc], alpha)
        # print('bool_value_update:', bool_value_update)
        '''更新气象相关参数(目前的数据无需更新)'''
        # k, C0 = 0.4, 3.0
        calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
        sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
        sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
        epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
        U_z = calculateBasicIndex.calculateWD()  # 20221123
        U_aver = U_z
        V_aver, W_aver = 0, 0
        '''保存当前时刻计算得到的x_wc, y_wc, z, u, v, w'''
        # ['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
        # 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario']
        temp_list = [start_time - pd.Timedelta(t_update, unit='s'), t_update, sampler_id, part_id, x_wc, y_wc, z,
                     u, v, w, U_aver_update, a_u, a_v, a_w, sigma_u, sigma_v, sigma_w, epsilon, secn_type]
        result_list.append(temp_list)  # 记录更新时刻
        # csvwrite.writerow(temp_list) 0214
        # count_nums = count_nums + 1  # 20230131
    result_df = pd.DataFrame(np.array(result_list),
                             columns=['时间', 't_update(视释放时刻为0时刻)', '采样点ID', '粒子ID', 'x_wc', 'y_wc', 'z',
                                      'u', 'v', 'w', 'U_aver', 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w',
                                      'epsilon', 'scenario'])
    # sample_conc_dic 存储单个粒子对于全部采样点的浓度贡献
    # result_df 存储单个粒子的轨迹
    # return sample_conc_dic, sample_partcount_dic, result_df
    # return sample_conc_dic, sample_w_dic, result_df
    return sample_conc_dic, sample_w_dic


'''并行的计算每个源每个粒子的轨迹'''
def backwardInNCBLParallel(sampling_point_df, z_s, Q_s, meteoro_list, N, start_time, dt_min, traj_file_path,
                           conc1_file_path, conc2_file_path, conc3_file_path, count_file_path, process_nums):
    secn_type = 'NCBL'
    particle_array = allocatParticles_b(sampling_point_df, N)
    print('particle_array:', particle_array)
    '''所有并行的粒子轨迹结果均写入下述文件中''' #0214
    # f = open(traj_file_path, 'w', encoding='utf-8-sig', newline='')
    # csvwrite = csv.writer(f)
    # csvwrite.writerow(['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
    #                    'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario'])
    sampler_point_xyz = sampling_point_df[['x_wc', 'y_wc', 'z']].values
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    k, C0 = 0.4, 3.0
    '''计算排放面源处的mu_w'''
    calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z_s, Z_i, z_0, L, secn_type, k=k, C0=C0)
    sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
    sigma_w = math.pow(sigma_w_s, 1 / 2)
    '''计算面源处的mu_w，与x，y无关'''

    '''构建sampler_part_id_list'''
    sampler_part_id_list = []
    N_si_list = []
    part_coor_b_list = []
    for i in range(len(sampling_point_df)):
        part_nums = particle_array[i]
        for j in range(part_nums):
            temp_list = [i, j]
            sampler_part_id_list.append(temp_list)
            N_si_list.append(part_nums)
            part_coor_b_list.append(list(sampler_point_xyz[i]))

    task_nums = len(sampler_part_id_list)
    print('task_nums:', task_nums)

    # mu_w, z_s, Q_s = emiss_info_list
    emiss_info_list_list = [[sigma_w, z_s, Q_s] for _ in range(task_nums)]
    traj_file_path_list = [traj_file_path for _ in range(task_nums)]
    sampling_point_df_list = [sampling_point_df for _ in range(task_nums)]
    meteoro_list_list = [meteoro_list for _ in range(task_nums)]
    start_time_list = [start_time for _ in range(task_nums)]
    dt_min_list = [dt_min for _ in range(task_nums)]
    cores = mp.cpu_count()
    print('cores:', cores)
    pool = mp.ProcessingPool(processes=process_nums)
    result_list = p_map(singlePartTraj_BackwardInNCBL, traj_file_path_list, emiss_info_list_list, sampler_part_id_list,
                        N_si_list, part_coor_b_list, sampling_point_df_list,
                        meteoro_list_list, start_time_list, dt_min_list)

    # pool = Pool(processes=process_nums)
    # result_list = pool.map(singlePartTraj_ForwardInNCBL, traj_file_path_list, emission_part_id_list,
    #                        Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list,
    #                        meteoro_list_list, start_time_list, dt_min_list)
    # result_list = list(pool.imap(singlePartTraj_ForwardInNCBL, traj_file_path_list, emission_part_id_list,
    #                              Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list, meteoro_list_list,
    #                              start_time_list, dt_min_list))

    # args_list = zip(traj_file_path_list, emission_part_id_list,Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list, meteoro_list_list, start_time_list, dt_min_list)
    # result_list = pool.starmap_async(singlePartTraj_ForwardInNCBL, args_list).get()
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程退出
    # f.close()
    # print(result_list)
    print(len(result_list))
    dic_list = []
    dic_value_list = []
    merged_w_dic = {}  # 将同一个采样点处的w合并到一个列表
    partcount_dic_list = []
    partcount_dic_value_list = []
    df_list = []

    '''6月7日改进数据处理，利用生成式列表'''
    # 待进行测试
    '''
    df_list = [item[2] for item in result_list]  # 轨迹
    dic_list = [item[0] for item in result_list]
    dic_value_list = [list((item[0]).values()) for item in result_list]  # 浓度
    partcount_dic_list = [item[1] for item in result_list]
    partcount_dic_value_list = [list((item[1]).values()) for item in result_list]  # 粒子数
    '''
    '''可以针对这个进行改进'''

    for i in range(task_nums):
        dic = result_list[i][0]
        dic_list.append(dic)
        dic_value_list.append(list(dic.values()))
        # df = result_list[i][2]
        # df_list.append(df)
        w_dic = result_list[i][1]
        for key, value in w_dic.items():
            if key in merged_w_dic:
                merged_w_dic[key].extend(value)
            else:
                merged_w_dic[key] = value

    ''' # 针对单个返回
    for i in range(task_nums):
        dic = result_list[i]
        # sample_partcount_dic = result_list[i][1]
        # df = result_list[i][2]
        dic_list.append(dic)
        dic_value_list.append(list(dic.values()))
        # partcount_dic_list.append(sample_partcount_dic)
        # partcount_dic_value_list.append(list(sample_partcount_dic.values()))
        # df_list.append(df)
    # '''

    '''计算溯源部分浓度2、浓度3'''
    # N为各采样点释放的粒子数列表
    e_A = 300  # 30*10，源面积
    # e_A = 30  # 20231207
    e_Q = Q_s # 1g/s
    conc1_dic = {}
    conc2_dic = {}
    for key, value in merged_w_dic.items():
        if len(value) == 0:
            conc1_dic[key] = 0
            conc2_dic[key] = 0
        else:
            # conc2_dic[key] = 1 / (N[int(key)]) * e_Q / e_A * (sum([2/abs(_) for _ in value]))  # 仅适用单个源
            # conc3_dic[key] = 1 / (N[int(key)]) * e_Q / e_A * (len(value) ** 2 / (sum([abs(_) for _ in value])))  # 仅适用单个源
            conc1_dic[key] = 1 / N * e_Q / e_A * (sum([2 / abs(_) for _ in value]))  # 仅适用单个源
            # conc2_dic[key] = 1 / N * e_Q / e_A * (len(value) ** 2 / (sum([abs(_) for _ in value])))  # 仅适用单个源
            conc2_dic[key] = 2 * 1 / N * e_Q / e_A * (len(value) ** 2 / (sum([abs(_) for _ in value])))  # 仅适用单个源，乘以2，20231204

    dic_total_value_list = list(np.array(dic_value_list).sum(axis=0))
    # partcount_dic_total_value_list = list(np.array(partcount_dic_value_list).sum(axis=0))
    # print(dic_total_value_list)
    # print(len(dic_total_value_list))
    # exit()
    '''合并浓度, 合并粒子轨迹结果'''
    print('--------------------------')
    dic_total = dict([(str(_), float(dic_total_value_list[_])) for _ in range(len(dic_total_value_list))])  # 记录该粒子对于各采样点的浓度贡献
    js = json.dumps(dic_total)
    f = open(conc3_file_path, 'w')
    f.write(js)
    f.close()

    '''20231117最后结果浓度需*2，计算部分代码未做修改，展示时做了修改'''
    js = json.dumps(conc1_dic)
    f = open(conc1_file_path, 'w')
    f.write(js)
    f.close()

    js = json.dumps(conc2_dic)
    f = open(conc2_file_path, 'w')
    f.write(js)
    f.close()



    # partcount_dic_total = dict(
    #     [(str(_), float(partcount_dic_total_value_list[_])) for _ in range(len(partcount_dic_total_value_list))])
    js = json.dumps(merged_w_dic)
    f = open(count_file_path, 'w')
    f.write(js)
    f.close()

    # df_total = pd.concat(df_list, axis=0)
    # df_total.to_csv(traj_file_path, encoding='utf-8-sig') #0214

    return


'''并行计算每个源每个粒子释放情况'''
'''计算对流情形下单个粒子的轨迹，将其视为一个子任务'''

def singlePartTraj_BackwardInCBL(traj_file_path, emiss_info_list, sampler_part_id, N_si, part_coor_b, sampling_point_df,meteoro_list, start_time, dt_min):
    # with open(traj_file_path, 'a', encoding='utf-8-sig', newline='') as f:
    np.random.seed()
    # f = open(traj_file_path, 'a', encoding='utf-8-sig', newline='') # 0214
    # csvwrite = csv.writer(f) # 0214
    mu_w, z_s, Q_s = emiss_info_list
    sampler_id, part_id = sampler_part_id
    print('sampler_id:', sampler_id, 'part_id:', part_id)
    secn_type = 'CBL'
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    # result_list = []
    # 记录回溯粒子（即从某排放源释放的粒子）对于各采样点的浓度贡献
    sample_conc_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])
    # 用于计算浓度方式2、方式3
    sample_w_dic = dict([(str(_), []) for _ in range(len(sampling_point_df))])
    # 记录各采样点i回溯到各源j处的粒子数
    # sample_partcount_dic = dict([(str(i) + '-' + str(j), 0) for i in range(len(sampling_point_df)) for j in range(len(emiss_df))])
    x_wc, y_wc, z = part_coor_b  # 粒子释放的初始坐标及采样点位置
    k, C0 = 0.4, 3.0
    calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
    sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
    sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
    pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
    epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
    U_z = calculateBasicIndex.calculateWD()  # 20221123
    U_aver = U_z
    V_aver, W_aver = 0, 0
    '''速度初始化20221116, 初始化时未考虑粒子反射'''
    U, V, W = initParticleV_inCBL(sigma_u, sigma_v, sigma_w, U_aver, V_aver, W_aver, u_star, w_star, z, Z_i)
    '''速度初始化20221116, 初始化时未考虑粒子反射'''

    t_update = 0
    bool_value_update, x_lgc, y_lgc = whetherInTargetArea_b([x_wc, y_wc], alpha)
    temp_list = [start_time, t_update, sampler_id, part_id, x_wc, y_wc, z,
                 U, V, W, U_aver, '', '', '', sigma_u, sigma_v, sigma_w,
                 '', '', '', '', '', '', '', '', '', '', '',
                 '', '', epsilon, secn_type]
    # csvwrite.writerow(temp_list) 0214
    # result_list.append(temp_list)  # 记录初始时刻

    another_bool_value_update = True

    while (bool_value_update & another_bool_value_update):  # 有一个不满足则停止
        # try:
        U_aver_update = U_aver  # 暂时简单化处理1127需确认
        V_aver_update, W_aver_update = V_aver, W_aver
        # 记录前一时刻的坐标
        x_wc_old, y_wc_old, z_old = x_wc, y_wc, z
        '''根据应用场景计算x,y,z,u,v,w'''
        calculateInCBL = CalculateInCBL(u_star=u_star, w_star=w_star, x_wc=x_wc, y_wc=y_wc, z=z,
                                        Z_i=Z_i, U=U, V=V, W=W,
                                        U_aver=U_aver_update, V_aver=V_aver_update, W_aver=W_aver_update,
                                        sigma_u=sigma_u, sigma_v=sigma_v, sigma_w=sigma_w,
                                        epsilon=epsilon, pd_U_aver_to_z=pd_U_aver_to_z,
                                        dt_min=dt_min, C0=C0)
        dt = calculateInCBL.dt
        t_update = t_update + dt
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, \
        w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, x_wc, y_wc, z = calculateInCBL.updateXYZ_b()
        # y_wc = 0 # 20231013,20240305
        '''遍历采样点，判断粒子是否撞击了到了采样点，计算采样点浓度'''
        '''11月9日需要进行确认'''
        e_A = 300  # 30*10，源面积 20240301
        # e_A = 30 # 20231117
        e_Rs = math.pow(e_A / math.pi, 1 / 2)
        e_mu_w = mu_w  # 提前计算好排放源处的sigma_w
        e_z = z_s
        e_Q = Q_s
        # 20231013
        '''
        if (x_lgc>=-30) and (x_lgc<=0): #去掉与y相关的判断
            if (z <= e_z) and (z_old > e_z):
                sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1/N_si*e_Q/e_A*math.pow(2*math.pi, 1/2)/e_mu_w
                sample_w_dic[str(sampler_id)].append(W)
        if x_lgc <= -31:
            another_bool_value_update = False  # 判断是否离开
        '''
        # '''
        if (x_lgc>=-30) and (x_lgc<=0) and (y_lgc>=-5) and (y_lgc<=5):
            if (z <= e_z) and (z_old > e_z):
                sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1/N_si*e_Q/e_A*math.pow(2*math.pi, 1/2)/e_mu_w
                sample_w_dic[str(sampler_id)].append(W)
        if x_lgc <= -31:
            another_bool_value_update = False  # 判断是否离开
        # '''
        '''
        if (z > e_z) and (z_old <= e_z) and (x_lgc >= -30) and (x_lgc <= 0) and (y_lgc >= -5) and (y_lgc <= 5):
            X = x_wc + (x_wc_old - x_wc) / (z_old - z) * (e_z - z)
            Y = y_wc + (y_wc_old - y_wc) / (z_old - z) * (e_z - z)
            # 不允许粒子反复贡献
            if X >= -e_Rs:
                sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1 / N_si * e_Q / e_A * math.pow(
                    2 * math.pi, 1 / 2) / e_mu_w
                #用于计算公式2、3，保存撞击到源上各个粒子的w
                sample_w_dic[str(sampler_id)].append(W)
                # sample_partcount_dic[str(sampler_id)+'-'+str(m)] = sample_partcount_dic[str(sampler_id)+'-'+str(m)]+1
            if X < -e_Rs:
                another_bool_value_update = False  # 判断是否离开
            #允许粒子反复贡献
            if determine_value <= e_Rs:
                sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1 / N_si * e_Q / e_A * math.pow(2 * math.pi, 1 / 2) / e_mu_w
                sample_partcount_dic[str(sampler_id) + '-' + str(m)] = sample_partcount_dic[str(sampler_id) + '-' + str(m)] + 1
                # sample_partcount_dic[str(m)] = sample_partcount_dic[str(m)] + 1
            #'''

        '''判断粒子是否需要进行反射，不改变v'''
        # print('反射前z, z_0, z_i:', z, z_0, z_i)
        z_r = max(z_0, 0.05)
        if z < z_r:
            z = 2 * z_r - z
            U, W = -U + 2 * U_aver, -W + 2 * W_aver
        if z > Z_i:
            z = 2 * Z_i - z
            U, W = -U + 2 * U_aver, -W + 2 * W_aver
        # print('反射后z, z_0, z_i:', z, z_0, z_i)
        # print(' ')
        '''判断粒子是否需要进行反射'''

        # csvwrite.writerow('#')
        bool_value_update, x_lgc, y_lgc = whetherInTargetArea_b([x_wc, y_wc], alpha)
        # print('bool_value_update:', bool_value_update)
        '''更新气象相关参数(目前的数据无需更新)'''
        # k, C0 = 0.4, 3
        # calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
        calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0) #20230705
        sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
        sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
        pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
        epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
        U_z = calculateBasicIndex.calculateWD()  # 20221123
        U_aver = U_z
        V_aver, W_aver = 0, 0
        '''保存当前时刻计算得到的x_wc, y_wc, z, u, v, w'''
        # ['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
        # 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario']
        temp_list = [start_time - pd.Timedelta(t_update, unit='s'), t_update, sampler_id, part_id, x_wc, y_wc, z,
                     U, V, W, U_aver_update, a_U, a_V, a_W, sigma_u, sigma_v, sigma_w,
                     g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1,
                     P_B_part2, w_A_aver, w_B_aver,
                     epsilon, secn_type]
        # result_list.append(temp_list)  # 记录更新时刻
        # csvwrite.writerow(temp_list) #0214
        # count_nums = count_nums+1  # 20230131
        # except Exception as e:
        #     print(e)
        #     break
    # result_df = pd.DataFrame(np.array(result_list),
    #                          columns=['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z',
    #                                   'U', 'V', 'W', 'U_aver',
    #                                   'a_U', 'a_V', 'a_W', 'sigma_u', 'sigma_v', 'sigma_w',
    #                                   'g_w', 'g_u', 'g_a', 'A', 'B', 'P_A', 'P_A_part1', 'P_A_part2', 'P_B', 'P_B_part1', 'P_B_part2',
    #                                   'w_A_aver/sigma_A', 'w_B_aver/sigma_B',
    #                                   'epsilon', 'scenario'])
    # sample_conc_dic 存储单个粒子对于全部采样点的浓度贡献
    # result_df 存储单个粒子的轨迹
    # return sample_conc_dic, sample_w_dic, result_df
    return sample_conc_dic, sample_w_dic



def singlePartTraj_BackwardInCBLv2(traj_file_path, emiss_info_list, sampler_part_id, N_si, part_coor_b, sampling_point_df,meteoro_list, start_time, dt_min):
    # with open(traj_file_path, 'a', encoding='utf-8-sig', newline='') as f:
    np.random.seed()
    # f = open(traj_file_path, 'a', encoding='utf-8-sig', newline='') # 0214
    # csvwrite = csv.writer(f) # 0214
    mu_w, z_s, Q_s = emiss_info_list
    sampler_id, part_id = sampler_part_id
    print('sampler_id:', sampler_id, 'part_id:', part_id)
    secn_type = 'CBL'
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    # result_list = []
    # 记录回溯粒子（即从某排放源释放的粒子）对于各采样点的浓度贡献
    sample_conc_dic = dict([(str(_), 0) for _ in range(len(sampling_point_df))])
    # 用于计算浓度方式2、方式3
    sample_w_dic = dict([(str(_), []) for _ in range(len(sampling_point_df))])
    # 记录各采样点i回溯到各源j处的粒子数
    # sample_partcount_dic = dict([(str(i) + '-' + str(j), 0) for i in range(len(sampling_point_df)) for j in range(len(emiss_df))])
    x_wc, y_wc, z = part_coor_b  # 粒子释放的初始坐标及采样点位置
    k, C0 = 0.4, 3.0
    calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
    sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
    sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
    pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
    epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
    U_z = calculateBasicIndex.calculateWD()  # 20221123
    U_aver = U_z
    V_aver, W_aver = 0, 0
    '''速度初始化20221116, 初始化时未考虑粒子反射'''
    U, V, W = initParticleV_inCBL(sigma_u, sigma_v, sigma_w, U_aver, V_aver, W_aver, u_star, w_star, z, Z_i)
    '''速度初始化20221116, 初始化时未考虑粒子反射'''

    t_update = 0
    bool_value_update, x_lgc, y_lgc = whetherInTargetArea_b([x_wc, y_wc], alpha)
    temp_list = [start_time, t_update, sampler_id, part_id, x_wc, y_wc, z,
                 U, V, W, U_aver, '', '', '', sigma_u, sigma_v, sigma_w,
                 '', '', '', '', '', '', '', '', '', '', '',
                 '', '', epsilon, secn_type]
    # csvwrite.writerow(temp_list) 0214
    # result_list.append(temp_list)  # 记录初始时刻

    another_bool_value_update = True

    while (bool_value_update & another_bool_value_update):  # 有一个不满足则停止
        try:
            U_aver_update = U_aver  # 暂时简单化处理1127需确认
            V_aver_update, W_aver_update = V_aver, W_aver
            # 记录前一时刻的坐标
            x_wc_old, y_wc_old, z_old = x_wc, y_wc, z
            '''根据应用场景计算x,y,z,u,v,w'''
            calculateInCBL = CalculateInCBL(u_star=u_star, w_star=w_star, x_wc=x_wc, y_wc=y_wc, z=z,
                                            Z_i=Z_i, U=U, V=V, W=W,
                                            U_aver=U_aver_update, V_aver=V_aver_update, W_aver=W_aver_update,
                                            sigma_u=sigma_u, sigma_v=sigma_v, sigma_w=sigma_w,
                                            epsilon=epsilon, pd_U_aver_to_z=pd_U_aver_to_z,
                                            dt_min=dt_min, C0=C0)
            dt = calculateInCBL.dt
            t_update = t_update + dt
            g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, \
            w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, x_wc, y_wc, z = calculateInCBL.updateXYZ_b()
            # y_wc = 0 # 20231013,20240305
            '''遍历采样点，判断粒子是否撞击了到了采样点，计算采样点浓度'''
            '''11月9日需要进行确认'''
            e_A = 300  # 30*10，源面积 20240301
            # e_A = 30 # 20231117
            e_Rs = math.pow(e_A / math.pi, 1 / 2)
            e_mu_w = mu_w  # 提前计算好排放源处的sigma_w
            e_z = z_s
            e_Q = Q_s
            # 20231013
            '''
            if (x_lgc>=-30) and (x_lgc<=0): #去掉与y相关的判断
                if (z <= e_z) and (z_old > e_z):
                    sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1/N_si*e_Q/e_A*math.pow(2*math.pi, 1/2)/e_mu_w
                    sample_w_dic[str(sampler_id)].append(W)
            if x_lgc <= -31:
                another_bool_value_update = False  # 判断是否离开
            '''
            # '''
            if (x_lgc>=-30) and (x_lgc<=0) and (y_lgc>=-5) and (y_lgc<=5):
                if (z <= e_z) and (z_old > e_z):
                    sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1/N_si*e_Q/e_A*math.pow(2*math.pi, 1/2)/e_mu_w
                    sample_w_dic[str(sampler_id)].append(W)
            if x_lgc <= -31:
                another_bool_value_update = False  # 判断是否离开
            # '''
            '''
            if (z > e_z) and (z_old <= e_z) and (x_lgc >= -30) and (x_lgc <= 0) and (y_lgc >= -5) and (y_lgc <= 5):
                X = x_wc + (x_wc_old - x_wc) / (z_old - z) * (e_z - z)
                Y = y_wc + (y_wc_old - y_wc) / (z_old - z) * (e_z - z)
                # 不允许粒子反复贡献
                if X >= -e_Rs:
                    sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1 / N_si * e_Q / e_A * math.pow(
                        2 * math.pi, 1 / 2) / e_mu_w
                    #用于计算公式2、3，保存撞击到源上各个粒子的w
                    sample_w_dic[str(sampler_id)].append(W)
                    # sample_partcount_dic[str(sampler_id)+'-'+str(m)] = sample_partcount_dic[str(sampler_id)+'-'+str(m)]+1
                if X < -e_Rs:
                    another_bool_value_update = False  # 判断是否离开
                #允许粒子反复贡献
                if determine_value <= e_Rs:
                    sample_conc_dic[str(sampler_id)] = sample_conc_dic[str(sampler_id)] + 1 / N_si * e_Q / e_A * math.pow(2 * math.pi, 1 / 2) / e_mu_w
                    sample_partcount_dic[str(sampler_id) + '-' + str(m)] = sample_partcount_dic[str(sampler_id) + '-' + str(m)] + 1
                    # sample_partcount_dic[str(m)] = sample_partcount_dic[str(m)] + 1
                #'''

            '''判断粒子是否需要进行反射，不改变v'''
            # print('反射前z, z_0, z_i:', z, z_0, z_i)
            z_r = max(z_0, 0.05)
            if z < z_r:
                z = 2 * z_r - z
                U, W = -U + 2 * U_aver, -W + 2 * W_aver
            if z > Z_i:
                z = 2 * Z_i - z
                U, W = -U + 2 * U_aver, -W + 2 * W_aver
            # print('反射后z, z_0, z_i:', z, z_0, z_i)
            # print(' ')
            '''判断粒子是否需要进行反射'''

            # csvwrite.writerow('#')
            bool_value_update, x_lgc, y_lgc = whetherInTargetArea_b([x_wc, y_wc], alpha)
            # print('bool_value_update:', bool_value_update)
            '''更新气象相关参数(目前的数据无需更新)'''
            # k, C0 = 0.4, 3
            # calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0)
            calculateBasicIndex = CalculateBasicIndex(u_star, w_star, z, Z_i, z_0, L, secn_type, k=k, C0=C0) #20230705
            sigma_u_s, sigma_v_s, sigma_w_s = calculateBasicIndex.calculateSigmaSquare()
            sigma_u, sigma_v, sigma_w = math.pow(sigma_u_s, 1 / 2), math.pow(sigma_v_s, 1 / 2), math.pow(sigma_w_s, 1 / 2)
            pd_U_aver_to_z = calculateBasicIndex.calculateVerGradofWD()
            epsilon = calculateBasicIndex.calculateEpsilon(sigma_w_s=sigma_w_s)
            U_z = calculateBasicIndex.calculateWD()  # 20221123
            U_aver = U_z
            V_aver, W_aver = 0, 0
            '''保存当前时刻计算得到的x_wc, y_wc, z, u, v, w'''
            # ['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z', 'u', 'v', 'w', 'U_aver',
            # 'a_u', 'a_v', 'a_w', 'sigma_u', 'sigma_v', 'sigma_w', 'epsilon', 'scenario']
            temp_list = [start_time - pd.Timedelta(t_update, unit='s'), t_update, sampler_id, part_id, x_wc, y_wc, z,
                         U, V, W, U_aver_update, a_U, a_V, a_W, sigma_u, sigma_v, sigma_w,
                         g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1,
                         P_B_part2, w_A_aver, w_B_aver,
                         epsilon, secn_type]
            # result_list.append(temp_list)  # 记录更新时刻
            # csvwrite.writerow(temp_list) #0214
            # count_nums = count_nums+1  # 20230131
        except Exception as e:
            print(e)
            break
    # result_df = pd.DataFrame(np.array(result_list),
    #                          columns=['时间', 't_update(视释放时刻为0时刻)', '排放源ID', '粒子ID', 'x_wc', 'y_wc', 'z',
    #                                   'U', 'V', 'W', 'U_aver',
    #                                   'a_U', 'a_V', 'a_W', 'sigma_u', 'sigma_v', 'sigma_w',
    #                                   'g_w', 'g_u', 'g_a', 'A', 'B', 'P_A', 'P_A_part1', 'P_A_part2', 'P_B', 'P_B_part1', 'P_B_part2',
    #                                   'w_A_aver/sigma_A', 'w_B_aver/sigma_B',
    #                                   'epsilon', 'scenario'])
    # sample_conc_dic 存储单个粒子对于全部采样点的浓度贡献
    # result_df 存储单个粒子的轨迹
    # return sample_conc_dic, sample_w_dic, result_df
    return sample_conc_dic, sample_w_dic



'''并行的计算每个源每个粒子的轨迹'''
def backwardInCBLParallel(sampling_point_df, z_s, Q_s, meteoro_list, N, start_time, dt_min, traj_file_path,
                          conc1_file_path, conc2_file_path, conc3_file_path, count_file_path, process_nums):
    secn_type = 'CBL'
    particle_array = allocatParticles_b(sampling_point_df, N)
    print('particle_array:', particle_array)
    sampler_point_xyz = sampling_point_df[['x_wc', 'y_wc', 'z']].values
    L, u_star, w_star, Z_i, z_0, alpha = meteoro_list
    k, C0 = 0.4, 3.0
    '''计算排放源处的sigma_w'''
    Z = z_s / Z_i
    b1, b2, b3, b4 = 0.0020, 1.2, 0.333, 0.72
    w_square_aver = math.pow(w_star, 2) * math.pow((b1 + b2 * Z * (1 - Z) * (1 - b3 * Z)), 2 / 3)
    w_cube_aver = math.pow(w_star, 3) * b4 * Z * (1 - Z) * (1 - b3 * Z)
    w_B_aver = (math.pow(math.pow(w_cube_aver, 2) + 8 * math.pow(w_square_aver, 3), 1 / 2) - w_cube_aver) / (
            4 * w_square_aver)
    w_A_aver = w_square_aver / (2 * w_B_aver)
    A = w_B_aver / (w_A_aver + w_B_aver)
    B = 1 - A
    sigma_A, sigma_B = w_A_aver, w_B_aver
    mu_w = 1.462 * (A * sigma_A + B * sigma_B)  # 与x、y无关


    '''构建sampler_part_id_list'''
    sampler_part_id_list = []
    N_si_list = []
    part_coor_b_list = []
    for i in range(len(sampling_point_df)):
        part_nums = particle_array[i]
        for j in range(part_nums):
            temp_list = [i, j]
            sampler_part_id_list.append(temp_list)
            N_si_list.append(part_nums)
            part_coor_b_list.append(list(sampler_point_xyz[i]))

    task_nums = len(sampler_part_id_list)
    print('task_nums:', task_nums)
    emiss_info_list_list = [[mu_w, z_s, Q_s] for _ in range(task_nums)]
    traj_file_path_list = [traj_file_path for _ in range(task_nums)]
    sampling_point_df_list = [sampling_point_df for _ in range(task_nums)]
    meteoro_list_list = [meteoro_list for _ in range(task_nums)]
    start_time_list = [start_time for _ in range(task_nums)]
    dt_min_list = [dt_min for _ in range(task_nums)]
    cores = mp.cpu_count()
    print('cores:', cores)
    pool = mp.ProcessingPool(processes=process_nums)
    # traj_file_path, emiss_info_list, sampler_part_id, N_si, part_coor_b, sampling_point_df, emiss_df, meteoro_list, start_time, dt_min


    result_list = p_map(singlePartTraj_BackwardInCBL, traj_file_path_list,
                        emiss_info_list_list, sampler_part_id_list,
                        N_si_list, part_coor_b_list, sampling_point_df_list,
                        meteoro_list_list, start_time_list, dt_min_list)
    # 20230705
    # result_list = p_map(singlePartTraj_BackwardInCBLv2, traj_file_path_list,
    #                     emiss_info_list_list, sampler_part_id_list,
    #                     N_si_list, part_coor_b_list, sampling_point_df_list,
    #                     meteoro_list_list, start_time_list, dt_min_list)
    # print('len(result_list):', len(result_list))  # 返回任务数，即正常运行的粒子数

    # pool = Pool(processes=process_nums)
    # result_list = pool.map(singlePartTraj_ForwardInNCBL, traj_file_path_list, emission_part_id_list,
    #                        Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list,
    #                        meteoro_list_list, start_time_list, dt_min_list)
    # result_list = list(pool.imap(singlePartTraj_ForwardInNCBL, traj_file_path_list, emission_part_id_list,
    #                              Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list, meteoro_list_list,
    #                              start_time_list, dt_min_list))

    # args_list = zip(traj_file_path_list, emission_part_id_list,Q_ei_list, N_ei_list, part_coor_list, sampling_point_df_list, meteoro_list_list, start_time_list, dt_min_list)
    # result_list = pool.starmap_async(singlePartTraj_ForwardInNCBL, args_list).get()
    pool.close()  # 关闭进程池，不再接受新的进程
    pool.join()  # 主进程阻塞等待子进程退出
    # f.close()
    # print(result_list)
    print(len(result_list))
    dic_list = []
    dic_value_list = []
    merged_W_dic = {}  # 将同一个采样点处的w合并到一个列表
    partcount_dic_list = []
    partcount_dic_value_list = []
    # df_list = []

    '''6月7日改进数据处理，利用生成式列表'''
    '''待进行测试
    df_list = [item[2] for item in result_list]  # 轨迹
    dic_list = [item[0] for item in result_list]
    dic_value_list = [list((item[0]).values()) for item in result_list]  # 浓度
    partcount_dic_list = [item[1] for item in result_list]
    partcount_dic_value_list = [list((item[1]).values()) for item in result_list]  # 粒子数'''

    for i in range(task_nums):
        dic = result_list[i][0]
        w_dic = result_list[i][1]
        dic_list.append(dic)
        dic_value_list.append(list(dic.values()))
        # df = result_list[i][2]
        # df_list.append(df)
        for key, value in w_dic.items():
            if key in merged_W_dic:
                merged_W_dic[key].extend(value)
            else:
                merged_W_dic[key] = value
    '''
    for i in range(task_nums):
        dic = result_list[i]
        # sample_partcount_dic = result_list[i][1]
        # df = result_list[i][2]
        dic_list.append(dic)
        dic_value_list.append(list(dic.values()))
        # partcount_dic_list.append(sample_partcount_dic)
        # partcount_dic_value_list.append(list(sample_partcount_dic.values()))
        # df_list.append(df)
    # '''

    '''计算溯源部分浓度1、浓度2'''
    # N为各采样点释放的粒子数列表
    e_A = 300  # 30*10，源面积 20240301
    # e_A = 30 # 20231207
    e_Q = Q_s  # 1g/s
    conc1_dic = {}
    conc2_dic = {}
    for key, value in merged_W_dic.items():
        if len(value) == 0:
            conc1_dic[key] = 0
            conc2_dic[key] = 0
        else:
            # conc2_dic[key] = 1 / (N[int(key)]) * e_Q / e_A * (sum([2/abs(_) for _ in value]))  # 仅适用单个源
            # conc3_dic[key] = 1 / (N[int(key)]) * e_Q / e_A * (len(value) ** 2 / (sum([abs(_) for _ in value])))  # 仅适用单个源

            conc1_dic[key] = 1 / N * e_Q / e_A * (sum([2 / abs(_) for _ in value]))  # 仅适用单个源
            # conc2_dic[key] = 1 / N * e_Q / e_A * (len(value) ** 2 / (sum([abs(_) for _ in value])))  # 仅适用单个源
            conc2_dic[key] = 2*1 / N * e_Q / e_A * (len(value) ** 2 / (sum([abs(_) for _ in value])))  # 仅适用单个源，乘以2，20231204

    dic_total_value_list = list(np.array(dic_value_list).sum(axis=0))
    # partcount_dic_total_value_list = list(np.array(partcount_dic_value_list).sum(axis=0))
    # print(dic_total_value_list)
    # print(len(dic_total_value_list))
    # exit()
    '''合并浓度, 合并粒子轨迹结果'''
    print('--------------------------')
    dic_total = dict(
        [(str(_), float(dic_total_value_list[_])) for _ in range(len(dic_total_value_list))])  # 记录该粒子对于各采样点的浓度贡献
    js = json.dumps(dic_total)
    f = open(conc3_file_path, 'w')
    f.write(js)
    f.close()

    '''粒子浓度1'''
    js = json.dumps(conc1_dic)
    f = open(conc1_file_path, 'w')
    f.write(js)
    f.close()

    '''20231117最后结果浓度需*2，计算部分代码未做修改，展示时做了修改'''
    '''粒子浓度2'''
    js = json.dumps(conc2_dic)
    f = open(conc2_file_path, 'w')
    f.write(js)
    f.close()

    # partcount_dic_total = dict(
    #     [(str(_), float(partcount_dic_total_value_list[_])) for _ in range(len(partcount_dic_total_value_list))])
    js = json.dumps(merged_W_dic)
    f = open(count_file_path, 'w')
    f.write(js)
    f.close()

    # df_total = pd.concat(df_list, axis=0)
    # df_total.to_csv('v2_'+traj_file_path, encoding='utf-8-sig') #0214
    # df_total.to_csv(traj_file_path, encoding='utf-8-sig')  # 0214

    return



# 采样点，不用进行坐标转换都可
def loadSamplingPoint(alpha):
    coor_list = [[0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5],
                 [25,0,1], [25,0,2], [25,0,3], [25,0,4], [25,0,5],
                 [50,0,1], [50,0,2], [50,0,3], [50,0,4], [50,0,5]]

    # coor_list = [[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5],
    #              [50,0,0], [50,0,1], [50,0,2], [50,0,3], [50,0,4], [50,0,5],
    #              [100,0,0], [100,0,1], [100,0,2], [100,0,3], [100,0,4], [100,0,5]]

    # coor_list = [[50, 0, 0], [50, 0, 1], [50, 0, 2], [50, 0, 3], [50, 0, 4], [50, 0, 5],
    #              [100, 0, 0], [100, 0, 1], [100, 0, 2], [100, 0, 3], [100, 0, 4], [100, 0, 5]]

    df = pd.DataFrame(np.array(coor_list), columns=['x_lgc', 'y_lgc', 'z'])
    x_coor_array = df['x_lgc'].values
    y_coor_array = df['y_lgc'].values
    x_wc_list = []
    y_wc_list = []
    for x_lgc, y_lgc in zip(x_coor_array, y_coor_array):
        x_wc, y_wc = transferCoord(x_lgc, y_lgc, alpha)
        x_wc_list.append(x_wc)
        y_wc_list.append(y_wc)
    df['x_wc'] = x_wc_list
    df['y_wc'] = y_wc_list
    df = df[['x_lgc', 'y_lgc', 'x_wc', 'y_wc', 'z']]
    return df

'''面源释放的一万个粒子'''
'''源点随机初始化,采样1万个粒子'''
def loadEmissionSource(N, alpha, z_s=0.05, Qs=1):
    # 'x_lgc', 'y_lgc', 'x_wc', 'y_wc', 'z', 'Qs'
    x_lgc_list = (-30*np.random.uniform(0, 1, size=(1, N)))[0]
    y_lgc_list = (10*(np.random.uniform(0, 1, size=(1, N))-0.5))[0]  # 20240222
    # y_lgc_list = (0*(np.random.uniform(0, 1, size=(1, N))-0.5))[0] # 20231013 令y=0
    emiss_df = pd.DataFrame()
    emiss_df['x_lgc'] = x_lgc_list
    emiss_df['y_lgc'] = y_lgc_list
    xy_wc_list = []
    for x_lgc, y_lgc in zip(x_lgc_list, y_lgc_list):
        x_wc, y_wc = transferCoord(x_lgc, y_lgc, alpha)
        xy_wc_list.append([x_wc, y_wc])
    emiss_df[['x_wc', 'y_wc']] = np.array(xy_wc_list)
    emiss_df[['z', 'Qs']] = np.array([[z_s, Qs] for _ in range(N)])
    return emiss_df

'''给采样点分配粒子'''
def allocatParticles_b(sampling_point_df, N):
    N_array = N*np.ones(len(sampling_point_df))
    N_array = N_array.astype(int)
    return N_array


if __name__ == '__main__':
    '''扩散--forward：对流情景--CBL'''
    '''
    date = '20240429v15'
    result_folder = './areaPollution/forward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # mete_list = [-2, 0.2, 2, 1000, 0.01, 270]  # 不考虑风向
    mete_list = [-2.5, 0.2, 2, 1000, 0.01, 270] # 20240301
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    Scenario = 'CBL'
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.1
    N = 1000000  # 面源释放一万个粒子
    # N = 10000
    # N = 10
    emiss_df = loadEmissionSource(N=N, alpha=alpha)
    print('emiss_df:', emiss_df)
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()
    # emission_source_df, sampling_point_df, meteoro_list, N, start_time, dt_min, traj_file_path,
    # conc_file_path, count_file_path, process_nums
    forwardInCBLParallel(emission_source_df=emiss_df, sampling_point_df=sample_point_df, meteoro_list=mete_list,
                         N=N, start_time=start_time, dt_min=dt_min,
                         traj_file_path=result_folder + Scenario + '_traj_N' + str(
                             N) + '_' + date + '.csv',
                         conc_file_path=result_folder + Scenario + '_N' + str(
                             N) + '_conc_' + date + '.txt',
                         count_file_path=result_folder + Scenario + '_N' + str(
                             N) + '_count_' + date + '.txt',
                         process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''



    '''扩散--forward：对流情景--NCBL'''
    # '''
    date = '20240502_02'
    result_folder = './areaPollution/forward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    mete_list = [-2, 0.2, 2, 1000, 0.01, 270]  # 不考虑风向
    mete_list = [100, 0.2, 2, 200, 0.01, 270]  # 不考虑风向
    mete_list = [-2.5, 0.2, 2, 1000, 0.01, 270]
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    # L = 100 # 20240125
    # Z_i = 200
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    Scenario = 'NCBL'
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2024, 1, 1, 0, 0))
    dt_min = 0.1
    N = 1000000  # 面源释放一万个粒子
    # N = 10
    emiss_df = loadEmissionSource(N=N, alpha=alpha)
    print('emiss_df:', emiss_df)
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()
    # emission_source_df, sampling_point_df, meteoro_list, N, start_time, dt_min, traj_file_path,
    # conc_file_path, count_file_path, process_nums
    forwardInNCBLParallel(emission_source_df=emiss_df, sampling_point_df=sample_point_df, meteoro_list=mete_list,
                          N=N, start_time=start_time, dt_min=dt_min,
                          traj_file_path=result_folder + Scenario + '_traj_N' + str(
                              N) + '_' + date + '.csv',
                          conc_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_conc_' + date + '.txt',
                          count_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_count_' + date + '.txt',
                          process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''





    '''溯源--backward：非对流情景--NCBL'''
    # '''
    date = '20240502_1'
    result_folder = './areaPollution/backward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # 当alpha=180时，x_wc = y_lgc, y_wc=-x_lgc
    # mete_list = [-3, 0.3, 2.823, 1000, 0.01, 180]  # 不考虑风向
    # mete_list = [-3, 0.3, 2.823, 1000, 0.01, 270]  # 不考虑风向
    mete_list = [-2, 0.2, 2, 1000, 0.01, 270]  # 不考虑风向
    mete_list = [100, 0.2, 2, 200, 0.01, 270]  # 不考虑风向
    mete_list = [-2.5, 0.2, 2, 1000, 0.01, 270]
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    # L = 100 # 20240125
    # Z_i = 200 # 20240126
    z_s = 0.05  # 源高
    Q_s = 1  # 面源源强
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    Scenario = 'NCBL'
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2024, 1, 1, 0, 0))
    dt_min = 0.1
    N = 500000  # 采样点释放
    # N = 1
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()

    backwardInNCBLParallel(sampling_point_df=sample_point_df, z_s=z_s, Q_s=Q_s, meteoro_list=mete_list,
                          N=N, start_time=start_time, dt_min=dt_min,
                          traj_file_path=result_folder + Scenario + '_traj_N' + str(
                              N) + '_' + date + '.csv',
                          conc1_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb1_' + date + '.txt',
                          conc2_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb2_' + date + '.txt',
                          conc3_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb3_' + date + '.txt',
                          count_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_count_' + date + '.txt',
                          process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''


    #'''溯源--backward：对流情景--CBL
    date = '20240429v14'
    result_folder = './areaPollution/backward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # 当alpha=180时，x_wc = y_lgc, y_wc=-x_lgc
    # mete_list = [-3, 0.3, 2.823, 1000, 0.01, 180]  # 不考虑风向
    # mete_list = [-3, 0.3, 2.823, 1000, 0.01, 270]  # 不考虑风向
    mete_list = [-2, 0.2, 2, 1000, 0.01, 270]  # 不考虑风向
    mete_list = [-2.5, 0.2, 2, 1000, 0.01, 270] # 20240301
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    z_s = 0.05  # 源高
    Q_s = 1  # 面源源强
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    Scenario = 'CBL'
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.01
    N = 500000  # 采样点释放
    # N = 5
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()

    backwardInCBLParallel(sampling_point_df=sample_point_df, z_s=z_s, Q_s=Q_s, meteoro_list=mete_list,
                          N=N, start_time=start_time, dt_min=dt_min,
                          traj_file_path=result_folder + Scenario + '_traj_N' + str(
                              N) + '_' + date + '.csv',
                          conc1_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb1_' + date + '.txt',
                          conc2_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb2_' + date + '.txt',
                          conc3_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb3_' + date + '.txt',
                          count_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_count_' + date + '.txt',
                          process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''







    # 后向-CBL
    # '''溯源--backward：非对流情景--CBL
    date = '20230711v2'
    result_folder = './areaPollution/backward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # 当alpha=180时，x_wc = y_lgc, y_wc=-x_lgc
    # mete_list = [-3, 0.3, 2.823, 1000, 0.01, 180]  # 不考虑风向
    mete_list = [-3, 0.3, 2.823, 1000, 0.01, 270]  # 不考虑风向
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    z_s = 0.05  # 源高
    Q_s = 1  # 面源源强
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.01
    # N = 10
    N = 20000  # 采样点释放
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()

    backwardInCBLParallel(sampling_point_df=sample_point_df, z_s=z_s, Q_s=Q_s, meteoro_list=mete_list,
                          N=N, start_time=start_time, dt_min=dt_min,
                          traj_file_path=result_folder + Scenario + '_traj_N' + str(
                              N) + '_' + date + '.csv',
                          conc1_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb1_' + date + '.txt',
                          conc2_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb2_' + date + '.txt',
                          conc3_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_concb3_' + date + '.txt',
                          count_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_count_' + date + '.txt',
                          process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''



    # 后向-NCBL,L=-3
    '''溯源--backward：非对流情景--NCBL'''
    date = '20230706L-3'
    result_folder = './areaPollution/backward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # 当alpha=180时，x_wc = y_lgc, y_wc=-x_lgc
    # mete_list = [1000, 0.3, np.nan, 1000, 0.01, 270]  # 不考虑风向
    mete_list = [-3, 0.3, 2.823, 1000, 0.01, 270]  # 不考虑风向
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    z_s = 0.05  # 源高
    Q_s = 1  # 面源源强
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    Scenario = 'NCBL'
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.1
    # N = 10
    N = 20000  # 采样点释放
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()

    backwardInNCBLParallel(sampling_point_df=sample_point_df, z_s=z_s, Q_s=Q_s, meteoro_list=mete_list,
                           N=N, start_time=start_time, dt_min=dt_min,
                           traj_file_path=result_folder + Scenario + '_traj_N' + str(
                               N) + '_' + date + '.csv',
                           conc1_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_concb1_' + date + '.txt',
                           conc2_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_concb2_' + date + '.txt',
                           conc3_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_concb3_' + date + '.txt',
                           count_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_count_' + date + '.txt',
                           process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()


    # 后向-NCBL
    '''溯源--backward：非对流情景--NCBL'''
    date = '20230706L1000'
    result_folder = './areaPollution/backward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # 当alpha=180时，x_wc = y_lgc, y_wc=-x_lgc
    mete_list = [1000, 0.3, np.nan, 1000, 0.01, 270]  # 不考虑风向
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    z_s = 0.05  # 源高
    Q_s = 1  # 面源源强
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.1
    # N = 10
    N = 20000  # 采样点释放
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()

    backwardInNCBLParallel(sampling_point_df=sample_point_df, z_s=z_s, Q_s=Q_s, meteoro_list=mete_list,
                           N=N, start_time=start_time, dt_min=dt_min,
                           traj_file_path=result_folder + Scenario + '_traj_N' + str(
                               N) + '_' + date + '.csv',
                           conc1_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_concb1_' + date + '.txt',
                           conc2_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_concb2_' + date + '.txt',
                           conc3_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_concb3_' + date + '.txt',
                           count_file_path=result_folder + Scenario + '_N' + str(
                               N) + '_count_' + date + '.txt',
                           process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()






























    '''扩散--forward：对流情景--CBL'''
    '''
    date = '20230705'
    result_folder = './areaPollution/forward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    mete_list = [-3, 0.3, 2.823, 1000, 0.01, 270]  # 不考虑风向
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.1
    N = 1000000  # 面源释放一万个粒子
    # N = 1
    emiss_df = loadEmissionSource(N=N, alpha=alpha)
    print('emiss_df:', emiss_df)
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()
    # emission_source_df, sampling_point_df, meteoro_list, N, start_time, dt_min, traj_file_path,
    # conc_file_path, count_file_path, process_nums
    forwardInCBLParallel(emission_source_df=emiss_df, sampling_point_df=sample_point_df, meteoro_list=mete_list,
                         N=N, start_time=start_time, dt_min=dt_min,
                         traj_file_path=result_folder + Scenario + '_traj_N' + str(
                             N) + '_' + date + '.csv',
                         conc_file_path=result_folder + Scenario + '_N' + str(
                             N) + '_conc_' + date + '.txt',
                         count_file_path=result_folder + Scenario + '_N' + str(
                             N) + '_count_' + date + '.txt',
                         process_nums=64)
    end = time.time()
    print('total run time:', end - start) 
    exit() #'''

    '''扩散--forward：对流情景--NCBL'''
    '''
    date = '20230705'
    result_folder = './areaPollution/forward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    mete_list = [-3, 0.3, 2.823, 1000, 0.01, 270]  # 不考虑风向
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    Scenario = 'NCBL'
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.1
    N = 1000000  # 面源释放一万个粒子
    # N = 10
    emiss_df = loadEmissionSource(N=N, alpha=alpha)
    print('emiss_df:', emiss_df)
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()
    # emission_source_df, sampling_point_df, meteoro_list, N, start_time, dt_min, traj_file_path,
    # conc_file_path, count_file_path, process_nums

    forwardInNCBLParallel(emission_source_df=emiss_df, sampling_point_df=sample_point_df, meteoro_list=mete_list,
                          N=N, start_time=start_time, dt_min=dt_min,
                          traj_file_path=result_folder + Scenario + '_traj_N' + str(
                              N) + '_' + date + '.csv',
                          conc_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_conc_' + date + '.txt',
                          count_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_count_' + date + '.txt',
                          process_nums=48)

    end = time.time()
    print('total run time:', end - start)
    exit() #'''



    # 前向-NCBL
    '''扩散--forward：非对流情景--NCBL'''
    # '''
    date = '20230705L1000'
    result_folder = './areaPollution/forward/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    mete_list = [1000, 0.3, np.nan, 1000, 0.01, 270]  # 不考虑风向
    L, u_star, w_star, Z_i, z_0, alpha = mete_list
    Scenario = determineScenario(Z_i=Z_i, L=L, u_star=u_star, w_star=w_star)
    print('Scenario:', Scenario)
    start_time = pd.to_datetime(datetime.datetime(2023, 1, 1, 0, 0))
    dt_min = 0.1
    N = 1000000  # 面源释放一万个粒子
    # N = 1
    emiss_df = loadEmissionSource(N=N, alpha=alpha)
    print('emiss_df:', emiss_df)
    sample_point_df = loadSamplingPoint(alpha)
    print('sample_point_df:', sample_point_df)
    start = time.time()
    # emission_source_df, sampling_point_df, meteoro_list, N, start_time, dt_min, traj_file_path,
    # conc_file_path, count_file_path, process_nums
    forwardInNCBLParallel(emission_source_df=emiss_df, sampling_point_df=sample_point_df, meteoro_list=mete_list,
                          N=N, start_time=start_time, dt_min=dt_min,
                          traj_file_path=result_folder + Scenario + '_traj_N' + str(
                              N) + '_' + date + '.csv',
                          conc_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_conc_' + date + '.txt',
                          count_file_path=result_folder + Scenario + '_N' + str(
                              N) + '_count_' + date + '.txt',
                          process_nums=48)
    end = time.time()
    print('total run time:', end - start)
    exit()  # '''







