#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/02/15
# @Author  :
# @Site    :
# @File    : UpdateStatistics.py
# @Software: PyCharm

import math
import numpy as np

'''为解决CBL中前后向模型对等问题'''

'''
1、对指标U、V、W，x_wc、y_wc、z进行更新计算
2、使用时需对对流边界层和非对流边界层进行判断
'''

'''一、对流边界层'''
class CalculateInCBL(object):
    def __init__(self, u_star, w_star, x_wc, y_wc, z, Z_i,
                 U, V, W, U_aver, V_aver, W_aver, sigma_u, sigma_v, sigma_w, epsilon, pd_U_aver_to_z,
                 dt_min=0.1, C0=3.0):
        self.u_star = u_star
        self.w_star = w_star  # L<0时才有意义？？？
        self.x_wc = x_wc
        self.y_wc = y_wc
        self.z = z
        self.Z_i = Z_i
        self.Z = self.z/self.Z_i
        self.U = U
        self.V = V
        self.W = W
        self.U_aver = U_aver
        self.V_aver = V_aver
        self.W_aver = W_aver
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        # self.sigma_min = min([self.sigma_u, self.sigma_v, self.sigma_w])
        self.sigma_u_s = math.pow(sigma_u, 2)
        self.sigma_v_s = math.pow(sigma_v, 2)
        self.sigma_w_s = math.pow(sigma_w, 2)
        # self.sigma_s = self.sigma_v_s*(self.sigma_u_s*self.sigma_w_s-math.pow(self.u_star, 4)) # 未计算使用到
        self.epsilon = epsilon
        self.pd_U_aver_to_z = pd_U_aver_to_z
        self.dt_min = dt_min
        self.C0 = C0
        # self.dt = 0.012 * self.sigma_w_s / (self.C0 * self.epsilon)
        self.dt = 0.05*self.sigma_w_s / (self.C0 * self.epsilon)  # 敏感性试验
        # self.dt = min(1.0, max(self.dt_min, self.dt))  # 1201 草场 20240316 扩散
        self.dt = min(0.01, max(self.dt_min, self.dt))  # 20230706 溯源测试0.01；面源扩散1.0/0.5
        # self.dt = min(1.5, max(self.dt_min, self.dt))  # 0208 水箱

    '''计算(w^2)̅和(w^3)̅'''
    def calculateW_2and3_aver(self):
        # print('self.z_i:', self.z_i)
        # b1, b2, b3, b4 = 0.0020, 1.2, 0.333, 0.72
        b1, b2, b3, b4, b5 = 0.0020, 1.2, 0.333, 0.72, 0.0007  # 20231009
        w_square_aver = math.pow(self.w_star, 2)*math.pow((b1+b2*self.Z*(1-self.Z)*(1-b3*self.Z)), 2/3)
        # w_cube_aver = math.pow(self.w_star, 3)*b4*self.Z*(1-self.Z)*(1-b3*self.Z)
        w_cube_aver = math.pow(self.w_star, 3) * (b5 + b4 * self.Z * (1 - self.Z) * (1 - b3 * self.Z))  #20231009
        return w_square_aver, w_cube_aver

    '''计算(w_A)̅和(w_B)̅'''
    def calculateW_AandB_aver(self):
        w_square_aver, w_cube_aver = self.calculateW_2and3_aver()
        w_B_aver = (math.pow(math.pow(w_cube_aver, 2)+8*math.pow(w_square_aver, 3), 1/2)-w_cube_aver)/(4*w_square_aver)
        w_A_aver = w_square_aver/(2*w_B_aver)
        return w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算P_A和P_B'''  #计算出来的P_A和P_B极小20221114
    def calculateP_AandB(self):
        w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateW_AandB_aver()
        sigma_A, sigma_B = w_A_aver, w_B_aver
        # P_A = 1/(math.pow(2*math.pi, 1/2)*sigma_A)*math.exp(-math.pow(self.W-w_A_aver, 2)/(2*math.pow(sigma_A, 2)))
        # P_B = 1/(math.pow(2*math.pi, 1/2)*sigma_B)*math.exp(-math.pow(self.W+w_B_aver, 2)/(2*math.pow(sigma_B, 2)))
        # 20230111
        P_A_part1 = 1/(math.pow(2*math.pi, 1/2)*sigma_A)
        P_A_part2 = math.exp(-math.pow(self.W-w_A_aver, 2)/(2*math.pow(sigma_A, 2)))
        P_B_part1 = 1/(math.pow(2*math.pi, 1/2)*sigma_B)
        P_B_part2 = math.exp(-math.pow(self.W+w_B_aver, 2)/(2*math.pow(sigma_B, 2)))
        P_A = P_A_part1*P_A_part2
        P_B = P_B_part1*P_B_part2

        # print('sigma_A/w_A_aver:', sigma_A, 'sigma_B/w_B_aver:', sigma_B)
        return P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算后向P_A和P_B'''
    def calculateP_AandB_b(self):
        w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateW_AandB_aver()
        sigma_A, sigma_B = w_A_aver, w_B_aver
        # P_A = 1/(math.pow(2*math.pi, 1/2)*sigma_A)*math.exp(-math.pow(self.W-w_A_aver, 2)/(2*math.pow(sigma_A, 2)))
        # P_B = 1/(math.pow(2*math.pi, 1/2)*sigma_B)*math.exp(-math.pow(self.W+w_B_aver, 2)/(2*math.pow(sigma_B, 2)))
        # 20230111
        '''改变P_A和P_B计算'''
        P_A_part1 = 1 / (math.pow(2 * math.pi, 1 / 2) * sigma_A)
        P_A_part2 = math.exp(-math.pow(self.W + w_A_aver, 2) / (2 * math.pow(sigma_A, 2)))
        P_B_part1 = 1 / (math.pow(2 * math.pi, 1 / 2) * sigma_B)
        P_B_part2 = math.exp(-math.pow(self.W - w_B_aver, 2) / (2 * math.pow(sigma_B, 2)))
        P_A = P_A_part1 * P_A_part2
        P_B = P_B_part1 * P_B_part2

        # print('sigma_A/w_A_aver:', sigma_A, 'sigma_B/w_B_aver:', sigma_B)
        return P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver


    '''计算A和B'''
    def calculateAandB(self):
        P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateP_AandB()
        A = w_B_aver/(w_A_aver+w_B_aver)
        B = w_A_aver/(w_A_aver+w_B_aver)
        # B = 1-A
        return A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算后向A和B'''
    def calculateAandB_b(self):
        P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateP_AandB_b()
        A = w_B_aver / (w_A_aver + w_B_aver)
        B = w_A_aver / (w_A_aver + w_B_aver)
        # B = 1-A
        return A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算I_1和I_2'''
    def calculateI_1and2(self): # SD calculateI_1and2_b is the same as this one.
        A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateAandB()
        # P_A, P_B, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateP_AandB()
        sigma_A, sigma_B = w_A_aver, w_B_aver
        I_1 = A/2*w_A_aver*(math.erf((self.W-w_A_aver)/(math.pow(2, 1/2)*sigma_A))+1)-A*math.pow(sigma_A, 2)*P_A
        I_2 = -B/2*w_B_aver*(math.erf((self.W+w_B_aver)/(math.pow(2, 1/2)*sigma_B))+1)-B*math.pow(sigma_B, 2)*P_B
        return I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算后向I_1和I_2'''
    def calculateI_1and2_b(self):  # SD calculateI_1and2_b is the same as this one.
        A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateAandB_b()
        # P_A, P_B, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateP_AandB()
        sigma_A, sigma_B = w_A_aver, w_B_aver
        I_1 = A / 2 * w_A_aver * (math.erf((self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A)) + 1) - A * math.pow(
            sigma_A, 2) * P_A
        I_2 = -B / 2 * w_B_aver * (math.erf((self.W + w_B_aver) / (math.pow(2, 1 / 2) * sigma_B)) + 1) - B * math.pow(
            sigma_B, 2) * P_B
        return I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算g_w,g_u,g_a'''
    def calculateG_wua(self):
        I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateI_1and2()
        # P_A, P_B, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateP_AandB()
        g_w = A*P_A+B*P_B   # 是否需要用sigma_w的计算公式进行替代
        g_u = 1/(math.pow(2*math.pi, 1/2)*self.sigma_u)*math.exp(-math.pow(self.U-self.U_aver, 2)/(2*self.sigma_u_s))
        g_a = g_u*g_w
        # print('A:', A, 'B:', B, 'P_A:', P_A, 'P_B:', P_B, 'g_w:', g_w, 'g_u:', g_u, 'g_a:', g_a)
        return g_w, g_u, g_a, I_1, I_2, A, B,  P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算后向g_w,g_u,g_a'''
    def calculateG_wua_b(self):
        I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateI_1and2_b()
        # P_A, P_B, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateP_AandB()
        g_w = A * P_A + B * P_B  # 是否需要用sigma_w的计算公式进行替代
        g_u = 1 / (math.pow(2 * math.pi, 1 / 2) * self.sigma_u) * math.exp(
            -math.pow(self.U - self.U_aver, 2) / (2 * self.sigma_u_s))
        g_a = g_u * g_w
        # g_a = A * P_A + B * P_B # 待确认
        # print('A:', A, 'B:', B, 'P_A:', P_A, 'P_B:', P_B, 'g_w:', g_w, 'g_u:', g_u, 'g_a:', g_a)
        return g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算ϕ_LB，ϕ_w'''
    def calculatePhi_LBandw(self):  # SD calculatePhi_LBandw_b is the same as this one.
        g_w, g_u, g_a, I_1, I_2, A, B,  P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateG_wua()
        # P_A, P_B, w_A_aver, w_B_aver, w_square_aver, w_cube_aver, A, B = self.calculateAandB()
        sigma_A = w_A_aver
        sigma_B = w_B_aver
        pd_sigma_u_s_to_z = 0 # 1127待确认？
        b1, b2, b3, b4 = 0.0020, 1.2, 0.333, 0.72
        pd_w_square_aver_to_z = 2*w_square_aver/(3*self.Z_i)*(b2*(1-self.Z)*(1-b3*self.Z)-b2*self.Z*(1-b3*self.Z)-b2*b3*self.Z*(1-self.Z))/(b1+b2*self.Z*(1-self.Z)*(1-b3*self.Z))
        pd_w_cube_aver_to_z = w_cube_aver/self.Z_i*((1-self.Z)*(1-b3*self.Z)-self.Z*(1-b3*self.Z)-b3*self.Z*(1-self.Z))/(self.Z*(1-self.Z)*(1-b3*self.Z))

        F = 4*w_square_aver*w_B_aver+w_cube_aver
        pd_w_B_aver_to_z = -1/F*(w_B_aver*pd_w_cube_aver_to_z+pd_w_square_aver_to_z*(w_B_aver*F/w_square_aver-3*w_square_aver))
        pd_sigma_B_to_z = pd_w_B_aver_to_z
        pd_w_A_aver_to_z = 1/(2*w_B_aver)*pd_w_square_aver_to_z-w_A_aver/w_B_aver*pd_w_B_aver_to_z
        pd_sigma_A_to_z = pd_w_A_aver_to_z
        pd_w_A_aver_to_x = 0 # 与x无关
        pd_w_B_aver_to_x = 0 # 与x无关
        pd_sigma_A_to_x = pd_w_A_aver_to_x
        pd_sigma_B_to_x = pd_w_B_aver_to_x

        '''20221109'''
        # pd_U_aver_to_z = ???  为calculateStatistics里的计算结果
        pd_A_to_z = 1 / (w_A_aver + w_B_aver) * (-A*pd_w_A_aver_to_z+B*pd_w_B_aver_to_z)
        pd_B_to_z = -pd_A_to_z
        pd_A_to_x, pd_B_to_x = 0, 0

        # phi_LB = -1 / 2 * (A * pd_w_A_aver_to_z + w_A_aver * pd_A_to_z) * math.erf(
        #     (self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A))
        # +sigma_A * (A * pd_sigma_A_to_z * (math.pow(self.W / sigma_A, 2) + 1) + sigma_A * pd_A_to_z) * P_A
        # +1 / 2 * (B * pd_w_B_aver_to_z + w_B_aver * pd_B_to_z) * math.erf(
        #     (self.W + w_B_aver) / (math.pow(2, 1 / 2) * sigma_B))
        # +sigma_B * (B * pd_sigma_B_to_z * (math.pow(self.W / sigma_B, 2) + 1) + sigma_B * pd_B_to_z) * P_B

        # if (self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A) >= 2.0 or (self.W + w_B_aver) / (
        #         math.pow(2, 1 / 2) * sigma_B) <= -2.0:
        #     phi_LB = sigma_A*(A*pd_sigma_A_to_z*(math.pow(self.W / sigma_A, 2) + 1)+sigma_A*pd_A_to_z)*P_A+\
        #              sigma_B*(B*pd_sigma_B_to_z*(math.pow(self.W / sigma_B, 2) + 1)+sigma_B*pd_B_to_z)*P_B

        phi_LB = 1/(2*math.pow(w_A_aver+w_B_aver, 2))*(math.pow(w_A_aver, 2)*pd_w_B_aver_to_z+math.pow(w_B_aver, 2)*pd_w_A_aver_to_z)*\
                 (math.erf((self.W+w_B_aver)/(math.pow(2, 1 / 2) * sigma_B))-math.erf((self.W-w_A_aver)/(math.pow(2, 1 / 2) * sigma_A)))+\
                 sigma_A*(A*pd_sigma_A_to_z*(math.pow(self.W / sigma_A, 2) + 1)+sigma_A*pd_A_to_z)*P_A+\
                     sigma_B*(B*pd_sigma_B_to_z*(math.pow(self.W / sigma_B, 2) + 1)+sigma_B*pd_B_to_z)*P_B

        ''' 20230227
        if (self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A) >= 2.5 or (self.W + w_B_aver) / (
                math.pow(2, 1 / 2) * sigma_B) <= -2.5:
            phi_LB = sigma_A*(A*pd_sigma_A_to_z*(math.pow(self.W / sigma_A, 2) + 1)+sigma_A*pd_A_to_z)*P_A+\
                     sigma_B*(B*pd_sigma_B_to_z*(math.pow(self.W / sigma_B, 2) + 1)+sigma_B*pd_B_to_z)*P_B
        '''

        # phi_w = g_a / g_w * phi_LB
        # 20240406
        phi_w = -1 / 2 * (A * pd_w_A_aver_to_z + w_A_aver * pd_A_to_z) * math.erf(
            (self.W - w_A_aver) / (math.pow(2, 1 / 2) * w_A_aver)) + \
                w_A_aver * (A * pd_w_A_aver_to_z * (math.pow(self.W / w_A_aver, 2) + 1) + w_A_aver * pd_A_to_z) * P_A + \
                1 / 2 * (B * pd_w_B_aver_to_z + w_B_aver * pd_B_to_z) * math.erf(
            (self.W + w_B_aver) / (math.pow(2, 1 / 2) * w_B_aver)) + \
                w_B_aver * (B * pd_w_B_aver_to_z * (math.pow(self.W / w_B_aver, 2) + 1) + w_B_aver * pd_B_to_z) * P_B
        # phi_w = 0  # 20240301
        return phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算后向ϕ_LB，ϕ_w'''
    def calculatePhi_LBandw_b(self):  # SD calculatePhi_LBandw_b is the same as this one.
        g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateG_wua_b()
        # P_A, P_B, w_A_aver, w_B_aver, w_square_aver, w_cube_aver, A, B = self.calculateAandB()
        sigma_A = w_A_aver
        sigma_B = w_B_aver
        pd_sigma_u_s_to_z = 0  # 1127待确认？
        b1, b2, b3, b4 = 0.0020, 1.2, 0.333, 0.72
        pd_w_square_aver_to_z = 2 * w_square_aver / (3 * self.Z_i) * (
                    b2 * (1 - self.Z) * (1 - b3 * self.Z) - b2 * self.Z * (1 - b3 * self.Z) - b2 * b3 * self.Z * (
                        1 - self.Z)) / (b1 + b2 * self.Z * (1 - self.Z) * (1 - b3 * self.Z))
        pd_w_cube_aver_to_z = w_cube_aver / self.Z_i * (
                    (1 - self.Z) * (1 - b3 * self.Z) - self.Z * (1 - b3 * self.Z) - b3 * self.Z * (1 - self.Z)) / (
                                          self.Z * (1 - self.Z) * (1 - b3 * self.Z))

        F = 4 * w_square_aver * w_B_aver + w_cube_aver
        pd_w_B_aver_to_z = -1 / F * (w_B_aver * pd_w_cube_aver_to_z + pd_w_square_aver_to_z * (
                    w_B_aver * F / w_square_aver - 3 * w_square_aver))
        pd_sigma_B_to_z = pd_w_B_aver_to_z
        pd_w_A_aver_to_z = 1 / (2 * w_B_aver) * pd_w_square_aver_to_z - w_A_aver / w_B_aver * pd_w_B_aver_to_z
        pd_sigma_A_to_z = pd_w_A_aver_to_z
        pd_w_A_aver_to_x = 0  # 与x无关
        pd_w_B_aver_to_x = 0  # 与x无关
        pd_sigma_A_to_x = pd_w_A_aver_to_x
        pd_sigma_B_to_x = pd_w_B_aver_to_x

        '''20221109'''
        # pd_U_aver_to_z = ???  为calculateStatistics里的计算结果
        pd_A_to_z = 1 / (w_A_aver + w_B_aver) * (-A * pd_w_A_aver_to_z + B * pd_w_B_aver_to_z)
        pd_B_to_z = -pd_A_to_z
        pd_A_to_x, pd_B_to_x = 0, 0

        # phi_LB = -1 / 2 * (A * pd_w_A_aver_to_z + w_A_aver * pd_A_to_z) * math.erf(
        #     (self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A))
        # +sigma_A * (A * pd_sigma_A_to_z * (math.pow(self.W / sigma_A, 2) + 1) + sigma_A * pd_A_to_z) * P_A
        # +1 / 2 * (B * pd_w_B_aver_to_z + w_B_aver * pd_B_to_z) * math.erf(
        #     (self.W + w_B_aver) / (math.pow(2, 1 / 2) * sigma_B))
        # +sigma_B * (B * pd_sigma_B_to_z * (math.pow(self.W / sigma_B, 2) + 1) + sigma_B * pd_B_to_z) * P_B

        # if (self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A) >= 2.0 or (self.W + w_B_aver) / (
        #         math.pow(2, 1 / 2) * sigma_B) <= -2.0:
        #     phi_LB = sigma_A*(A*pd_sigma_A_to_z*(math.pow(self.W / sigma_A, 2) + 1)+sigma_A*pd_A_to_z)*P_A+\
        #              sigma_B*(B*pd_sigma_B_to_z*(math.pow(self.W / sigma_B, 2) + 1)+sigma_B*pd_B_to_z)*P_B

        phi_LB = 1 / (2 * math.pow(w_A_aver + w_B_aver, 2)) * (
                    math.pow(w_A_aver, 2) * pd_w_B_aver_to_z + math.pow(w_B_aver, 2) * pd_w_A_aver_to_z) * \
                 (math.erf((self.W + w_B_aver) / (math.pow(2, 1 / 2) * sigma_B)) - math.erf(
                     (self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A))) + \
                 sigma_A * (A * pd_sigma_A_to_z * (math.pow(self.W / sigma_A, 2) + 1) + sigma_A * pd_A_to_z) * P_A + \
                 sigma_B * (B * pd_sigma_B_to_z * (math.pow(self.W / sigma_B, 2) + 1) + sigma_B * pd_B_to_z) * P_B

        ''' 20230227
        if (self.W - w_A_aver) / (math.pow(2, 1 / 2) * sigma_A) >= 2.5 or (self.W + w_B_aver) / (
                math.pow(2, 1 / 2) * sigma_B) <= -2.5:
            phi_LB = sigma_A*(A*pd_sigma_A_to_z*(math.pow(self.W / sigma_A, 2) + 1)+sigma_A*pd_A_to_z)*P_A+\
                     sigma_B*(B*pd_sigma_B_to_z*(math.pow(self.W / sigma_B, 2) + 1)+sigma_B*pd_B_to_z)*P_B
        '''

        # phi_w = g_a / g_w * phi_LB
        '''20231226、20240316、20240406gai'''
        phi_w = -1/2*(A*pd_w_A_aver_to_z+w_A_aver*pd_A_to_z)*math.erf((self.W-w_A_aver)/(math.pow(2, 1 / 2) * w_A_aver))+\
                w_A_aver*(A*pd_w_A_aver_to_z*(math.pow(self.W/w_A_aver, 2)+1)+w_A_aver*pd_A_to_z)*P_A+\
                1/2*(B*pd_w_B_aver_to_z+w_B_aver*pd_B_to_z)*math.erf((self.W+w_B_aver)/(math.pow(2, 1 / 2) * w_B_aver))+\
                w_B_aver*(B*pd_w_B_aver_to_z*(math.pow(self.W/w_B_aver, 2)+1)+w_B_aver*pd_B_to_z)*P_B
        # phi_w = 0  # 20240301
        return phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算ϕ_u'''
    def calculatePhi_u(self):   # SD calculatePhi_u_b is the same as this one
        '''对流边界层中phi_u=0'''
        phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculatePhi_LBandw()
        pd_sigma_u_s_to_x = 0 #????
        pd_sigma_u_s_to_z = 0
        # phi_u = g_a / 2 * pd_sigma_u_s_to_x * (1 + math.pow(self.U, 2) / self.sigma_u_s - self.U * self.U_aver / self.sigma_u_s)
        phi_u = self.W*g_a*(self.pd_U_aver_to_z+(self.U-self.U_aver)/(2*self.sigma_u_s)*pd_sigma_u_s_to_z)
        # 0711
        # phi_u = self.W*g_a*(-self.pd_U_aver_to_z+(self.U-self.U_aver)/(2*self.sigma_u_s)*pd_sigma_u_s_to_z)
        # phi_u = 0   # 20240301
        return phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2,  w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算后向ϕ_u'''
    def calculatePhi_u_b(self):  # SD calculatePhi_u_b is the same as this one
        '''对流边界层中phi_u=0'''
        phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculatePhi_LBandw_b()
        pd_sigma_u_s_to_x = 0  # ????
        pd_sigma_u_s_to_z = 0
        # phi_u = g_a / 2 * pd_sigma_u_s_to_x * (1 + math.pow(self.U, 2) / self.sigma_u_s - self.U * self.U_aver / self.sigma_u_s)
        phi_u = self.W * g_a * (self.pd_U_aver_to_z + (self.U - self.U_aver) / (2 * self.sigma_u_s) * pd_sigma_u_s_to_z)
        # phi_u = self.W*g_a*(-self.pd_U_aver_to_z + (self.U - self.U_aver) / (2 * self.sigma_u_s) * pd_sigma_u_s_to_z)
        # 0711
        # phi_u = self.W*g_a*(-self.pd_U_aver_to_z+(self.U-self.U_aver)/(2*self.sigma_u_s)*pd_sigma_u_s_to_z)
        # phi_u = 0  # 20240301
        return phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算Q'''
    def calculateQ(self):   # SD calculateQ_b is the same as this one
        phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2,  w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculatePhi_u()
        sigma_A_s = math.pow(w_A_aver, 2)
        sigma_B_s = math.pow(w_B_aver, 2)
        Q = A * (self.W - w_A_aver) * P_A / sigma_A_s + B * (self.W + w_B_aver) * P_B / sigma_B_s
        return Q, phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算后向Q'''
    def calculateQ_b(self):  # SD calculateQ_b is the same as this one
        phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculatePhi_u_b()
        sigma_A_s = math.pow(w_A_aver, 2)
        sigma_B_s = math.pow(w_B_aver, 2)
        '''改变Q'''
        Q = A * (self.W - w_A_aver) * P_A / sigma_A_s + B * (self.W + w_B_aver) * P_B / sigma_B_s  # 20240316
        return Q, phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver

    '''计算a_u, a_v, a_w'''
    def calculateA(self):
        Q, phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateQ()
        # phi_u,.... = self.calculatePhi_u()
        # phi_u = 0
        # print('g_w, g_u, g_a, Q, phi_w:', g_w, g_u, g_a, Q, phi_w)
        # a_V = -(self.C0 * self.epsilon / (2 * self.sigma_v_s)) * (self.V - self.V_aver) + 1 / 2 * (1 + math.pow(self.V, 2) / self.sigma_v_s-self.V*self.V_aver/self.sigma_v_s)
        a_V = -(self.C0 * self.epsilon / (2 * self.sigma_v_s)) * (self.V - self.V_aver)   # 1128
        # a_U = -(self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (self.U - self.U_aver) + phi_u / g_a
        # a_W = -(self.C0 * self.epsilon / (2 * g_w)) * Q + phi_w / g_a
        '''20231204，20240301，20240412'''
        a_U = -(self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (self.U - self.U_aver) + phi_u / g_a
        # a_W = -(self.C0 * self.epsilon / (2 * g_w)) * Q
        a_W = -(self.C0 * self.epsilon / (2 * g_w)) * Q + phi_w/g_w # 20240316
        if a_W > 3.0*self.C0*self.epsilon/self.sigma_w:
            a_W = 3.0*self.C0*self.epsilon/self.sigma_w
        if a_W < -3.0*self.C0*self.epsilon/self.sigma_w:
            a_W = -3.0*self.C0*self.epsilon/self.sigma_w
        '''20231219'''
        # a_U = -(self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (self.U - self.U_aver)
        # a_W = -(self.C0 * self.epsilon / (2 * g_w)) * Q + phi_w / g_w
        #  20230110
        # a_W = -(self.C0 * self.epsilon / (2 * g_w)) * Q + phi_w
        # print('a_U:', a_U, 'a_V:', a_V, 'a_W:', a_W)
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W

    '''计算后向a_u, a_v, a_w'''
    def calculateA_b(self):
        # 20240301
        Q, phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateQ() #S1
        # Q, phi_u, phi_LB, phi_w, g_w, g_u, g_a, I_1, I_2, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, w_square_aver, w_cube_aver = self.calculateQ_b() #S2
        # phi_u,.... = self.calculatePhi_u()
        # phi_u = 0
        # print('g_w, g_u, g_a, Q, phi_w:', g_w, g_u, g_a, Q, phi_w)
        # a_V = (self.C0 * self.epsilon / (2 * self.sigma_v_s)) * (self.V - self.V_aver) + 1 / 2 * (
        #             1 + math.pow(self.V, 2) / self.sigma_v_s-self.V*self.V_aver/self.sigma_v_s)
        '''
        a_V = (self.C0 * self.epsilon / (2 * self.sigma_v_s)) * (self.V - self.V_aver)  # 1128
        a_U = (self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (self.U - self.U_aver) + phi_u / g_a
        a_W = (self.C0 * self.epsilon / (2 * g_w)) * Q + phi_w / g_a
        '''
        '''3月30日修改,待确认，U、V形式其实和前向一致'''
        # a_V = (self.C0 * self.epsilon / (2 * self.sigma_v_s)) * (-(self.V - self.V_aver))  # 1128
        a_V = -self.C0 * self.epsilon / (2 * self.sigma_v_s) * self.V # 20240406
        #v1
        # a_U = (self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (-(self.U - self.U_aver)) - phi_u / g_a #SDSD multiplied 2nd term by (-1)
        # a_W = (self.C0 * self.epsilon / (2 * g_w)) * (-Q) - phi_w / g_a #SDSD multiplied 2nd term by (-1)
        #v2 1204之前使用
        # a_U = (self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (-(self.U - self.U_aver)) + phi_u / g_a  # SDSD multiplied 2nd term by (-1)
        # a_W = (self.C0 * self.epsilon / (2 * g_w)) * (-Q) + phi_w / g_a  # SDSD multiplied 2nd term by (-1)
        '''20231204，20240301，20240406，20240412'''
        a_U = -(self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (self.U - self.U_aver)+phi_u/g_a
        a_W = -(self.C0 * self.epsilon / (2 * g_w)) * Q - phi_w / g_w
        if a_W > 3.0*self.C0*self.epsilon/self.sigma_w:
            a_W = 3.0*self.C0*self.epsilon/self.sigma_w
        if a_W < -3.0*self.C0*self.epsilon/self.sigma_w:
            a_W = -3.0*self.C0*self.epsilon/self.sigma_w
        '''20231219'''
        # a_U = -(self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (self.U - self.U_aver)
        # a_W = -(self.C0 * self.epsilon / (2 * g_w)) * Q + phi_w / g_w
        # 20230706
        # a_U = (self.C0 * self.epsilon / (2 * self.sigma_u_s)) * (-(self.U - self.U_aver)) + phi_u / g_a  # SDSD multiplied 2nd term by (-1)
        # a_W = (self.C0 * self.epsilon / (2 * g_w)) * Q + phi_w / g_a
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W

    # ξ --> ksi
    '''计算du,dv,dw'''
    def calculateDeltaVelocity(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W = self.calculateA()
        '''dξu，dξw，dξv基于高斯分布生成'''
        delta_ksi_u, delta_ksi_v, delta_ksi_w = np.random.normal(loc=0.0, scale=math.pow(self.dt, 1/2), size=3)
        delta_U = a_U*self.dt+math.pow(self.C0*self.epsilon, 1/2)*delta_ksi_u
        delta_W = a_W*self.dt+math.pow(self.C0*self.epsilon, 1/2)*delta_ksi_w
        delta_V = a_V*self.dt+math.pow(self.C0*self.epsilon, 1/2)*delta_ksi_v
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, delta_U, delta_V, delta_W

    '''计算后向du,dv,dw'''
    def calculateDeltaVelocity_b(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W = self.calculateA_b()
        # print('a_U:', a_U, 'a_V:', a_V, 'a_W:', a_W)
        '''dξu，dξw，dξv基于高斯分布生成'''
        delta_ksi_u, delta_ksi_v, delta_ksi_w = np.random.normal(loc=0.0, scale=math.pow(self.dt, 1/2), size=3)
        delta_U = a_U * self.dt + math.pow(self.C0 * self.epsilon, 1 / 2) * delta_ksi_u # SD multiply the 1st term by (-1) #SDSD deleted -1
        delta_W = a_W * self.dt + math.pow(self.C0 * self.epsilon, 1 / 2) * delta_ksi_w # SD multiply the 1st term by (-1) #SDSD deleted -1
        delta_V = a_V * self.dt + math.pow(self.C0 * self.epsilon, 1 / 2) * delta_ksi_v # SD multiply the 1st term by (-1) #SDSD deleted -1
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, delta_U, delta_V, delta_W


    '''更新u，v，w'''
    def updateUVW(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, delta_U, delta_V, delta_W = self.calculateDeltaVelocity()
        U = self.U + delta_U
        V = self.V + delta_V
        W = self.W + delta_W
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W

    '''更新后向u，v，w'''
    def updateUVW_b(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, delta_U, delta_V, delta_W = self.calculateDeltaVelocity_b()
        U = self.U + delta_U
        V = self.V + delta_V
        W = self.W + delta_W
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W

    '''基于更新的u，v，w，计算dx，dy，dz'''
    def calculateDeltaDis(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W = self.updateUVW()
        # dt = 0.012 * math.pow(self.sigma_min, 2) / (self.C0 * self.epsilon)
        delta_x = U * self.dt
        delta_y = V * self.dt
        delta_z = W * self.dt
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, delta_x, delta_y, delta_z

    '''基于更新的后向u，v，w，计算dx，dy，dz'''
    def calculateDeltaDis_b(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W = self.updateUVW_b()
        # dt = 0.012*math.pow(self.sigma_min, 2)/(self.C0*self.epsilon)
        delta_x = -U * self.dt
        delta_y = -V * self.dt
        delta_z = -W * self.dt
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, delta_x, delta_y, delta_z

    '''更新x，y，z'''
    def updateXYZ(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, delta_x, delta_y, delta_z = self.calculateDeltaDis()
        x_wc = self.x_wc + delta_x
        y_wc = self.y_wc + delta_y
        z = self.z + delta_z
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, x_wc, y_wc, z

    def updateXYZ_b(self):
        g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, delta_x, delta_y, delta_z = self.calculateDeltaDis_b()
        x_wc = self.x_wc + delta_x
        y_wc = self.y_wc + delta_y
        z = self.z + delta_z
        return g_w, g_u, g_a, A, B, P_A, P_A_part1, P_A_part2, P_B, P_B_part1, P_B_part2, w_A_aver, w_B_aver, a_U, a_V, a_W, U, V, W, x_wc, y_wc, z



'''二、非对流边界层'''
class CalculateInNCBL(object):
    def __init__(self, u_star, w_star, x_wc, y_wc, z, Z_i, L,
                 u, v, w, U_aver_update, V_aver_update, W_aver_update,
                 sigma_u, sigma_v, sigma_w, dt_min, epsilon, C0=3.0):
        self.u_star = u_star
        self.w_star = w_star
        self.x_wc = x_wc
        self.y_wc = y_wc
        self.z = z
        self.Z_i = Z_i
        self.Z = self.z/self.Z_i
        self.L = L
        self.u = u
        self.v = v
        self.w = w
        # 假设为更新时刻的风速
        self.U_aver_update = U_aver_update
        self.V_aver_update = V_aver_update
        self.W_aver_update = W_aver_update
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.sigma_u_s = math.pow(sigma_u, 2)
        self.sigma_v_s = math.pow(sigma_v, 2)
        self.sigma_w_s = math.pow(sigma_w, 2)
        # self.sigma_s = self.sigma_v_s*(self.sigma_u_s*self.sigma_w_s-math.pow(self.u_star, 4))
        # 20230309更新
        self.sigma_s = self.sigma_v_s*(self.sigma_u_s*self.sigma_w_s-math.pow(self.u_star*math.pow(1-self.z/self.Z_i, 3/4), 4))
        self.dt_min = dt_min
        self.epsilon = epsilon
        self.C0 = C0
        # self.dt = 0.012 * self.sigma_w_s / (self.C0 * self.epsilon)  # 为gan生成数据用先
        self.dt = 0.05 * self.sigma_w_s / (self.C0 * self.epsilon) # 敏感性试验
        # print('原始dt:', self.dt)
        self.dt = min(0.2, max(self.dt_min, self.dt)) #1.0

    '''计算sigma_u_s、sigma_v_s、sigma_w_s关于z的偏导数'''
    def calculatePd(self):
        if self.L > 0:
            pd_sigma_u_s_to_z = 4.0 * math.pow(self.u_star, 2) * 3 / 2 * math.pow(1 - self.Z, 1 / 2) * (-1 / self.Z_i)
            pd_sigma_v_s_to_z = 5.0 * math.pow(self.u_star, 2) * 3 / 2 * math.pow(1 - self.Z, 1 / 2) * (-1 / self.Z_i)
            pd_sigma_w_s_to_z = 1.7 * math.pow(self.u_star, 2) * 3 / 2 * math.pow(1 - self.Z, 1 / 2) * (-1 / self.Z_i)  # 20230215
        else:
            if -self.Z_i/self.L < 1000:
                pd_sigma_v_s_to_z = -math.pow(self.u_star, 2)/self.Z_i
                pd_sigma_u_s_to_z = pd_sigma_v_s_to_z
                pd_sigma_w_s_to_z = 1.7*math.pow(self.u_star, 2)*math.pow(1-self.z/self.L, -1/3)*(-1/self.L)
            # 20230301改
            else:
                b1, b2, b3 = 0.0020, 1.2, 0.333
                pd_sigma_w_s_to_z = 2/3*math.pow(self.w_star, 2)*math.pow(b1+b2*self.Z*(1-self.Z)*(1-b3*self.Z), -1/3)*\
                                    b2*((1/self.Z_i-2*self.Z/self.Z_i)*(1-b3*self.Z)+(-b3/self.Z_i)*(self.Z*(1-self.Z)))
                pd_sigma_v_s_to_z = -math.pow(self.u_star, 2) / self.Z_i # 面源
                # pd_sigma_v_s_to_z = 0
                pd_sigma_u_s_to_z = pd_sigma_v_s_to_z
            # 20230301改
        return pd_sigma_u_s_to_z, pd_sigma_v_s_to_z, pd_sigma_w_s_to_z


    '''计算a_u, a_v, a_w'''
    def calculateA(self):
        # uw_aver = -math.pow(self.u_star, 2)
        # 20230315
        uw_aver = -math.pow(self.u_star, 2)*math.pow(1-self.Z, 2/3)
        vw_aver = 0
        uv_aver = 0
        # pd_uw_aver_to_z = 0
        # 20230315
        pd_uw_aver_to_z = 2/(3*self.Z_i)*math.pow(self.u_star, 2)*math.pow(1-self.Z, -1/3)
        pd_vw_aver_to_z = 0
        # print(self.z)
        # print(self.z_i)
        # print(self.u_star, self.z/self.z_i)
        pd_sigma_u_s_to_z, pd_sigma_v_s_to_z, pd_sigma_w_s_to_z = self.calculatePd()
        # print('self.W:', self.W)
        # print('self.sigma_min_s:', self.sigma_min_s)
        b = math.pow(self.C0*self.epsilon, 1/2)
        a_u = -math.pow(b, 2) / (2 * self.sigma_s) * (self.u * (self.sigma_v_s * self.sigma_w_s - math.pow(vw_aver,2))
        + self.v * uw_aver * vw_aver - self.w * self.sigma_v_s * uw_aver) + 1 / 2 * pd_uw_aver_to_z
        +self.u * self.w / (2 * self.sigma_s) * (pd_sigma_u_s_to_z * (self.sigma_v_s * self.sigma_w_s - math.pow(vw_aver, 2))
        - pd_uw_aver_to_z * self.sigma_v_s * uw_aver)
        +self.v * self.w / (2 * self.sigma_s) * (pd_sigma_u_s_to_z * uw_aver * vw_aver - pd_uw_aver_to_z * self.sigma_u_s * vw_aver)
        +math.pow(self.w, 2) / (2 * self.sigma_s) * (pd_uw_aver_to_z * self.sigma_u_s * self.sigma_v_s - pd_sigma_u_s_to_z * self.sigma_v_s * uw_aver)
        # '''
        a_v = -math.pow(b, 2) / (2 * self.sigma_s) * (self.u * uw_aver * vw_aver + self.v * (
                    self.sigma_u_s * self.sigma_w_s - math.pow(uw_aver, 2)) - self.w * self.sigma_u_s * vw_aver) + 1 / 2 * pd_vw_aver_to_z
        +self.u * self.w / (2 * self.sigma_s) * (pd_sigma_v_s_to_z * uw_aver * vw_aver - pd_vw_aver_to_z * self.sigma_v_s * uw_aver)
        +self.v * self.w / (2 * self.sigma_s) * (pd_sigma_v_s_to_z * (
                    self.sigma_u_s * self.sigma_w_s - math.pow(uw_aver, 2) - pd_vw_aver_to_z * self.sigma_u_s * vw_aver))
        +math.pow(self.w, 2) / (2 * self.sigma_s) * (
                    pd_vw_aver_to_z * self.sigma_u_s * self.sigma_v_s - pd_sigma_v_s_to_z * self.sigma_u_s * vw_aver)
        # '''
        # a_v = -self.C0*self.epsilon/(2*self.sigma_v_s)*self.v #20240222
        # '''
        a_w = math.pow(b, 2) / (2 * self.sigma_s) * (
                    self.u * self.sigma_v_s * uw_aver + self.v * self.sigma_u_s * vw_aver - self.w * self.sigma_u_s * self.sigma_v_s) + 1 / 2 * pd_sigma_w_s_to_z
        +self.u * self.w / (2 * self.sigma_s) * (pd_uw_aver_to_z * (self.sigma_v_s * self.sigma_w_s - math.pow(vw_aver,
                                                                                      2)) + pd_vw_aver_to_z * uw_aver * vw_aver - pd_sigma_w_s_to_z * self.sigma_v_s * uw_aver)
        +self.v * self.w / (2 * self.sigma_s) * (pd_uw_aver_to_z * uw_aver * vw_aver + pd_vw_aver_to_z * (
                    self.sigma_u_s * self.sigma_w_s - math.pow(uw_aver, 2) - pd_sigma_w_s_to_z * self.sigma_u_s * vw_aver))
        +math.pow(self.w, 2) / (2 * self.sigma_s) * (
                    pd_sigma_w_s_to_z * self.sigma_u_s * self.sigma_v_s - pd_uw_aver_to_z * self.sigma_v_s * uw_aver - pd_vw_aver_to_z * self.sigma_u_s * vw_aver)
        # '''
        # a_w = -self.C0*self.epsilon/(2*self.sigma_w_s)*self.w+1/2*pd_sigma_w_s_to_z*(1+math.pow(self.w/self.sigma_w, 2)) # 20240102
        # a_u, a_v = 0, 0 # 20240125
        # a_u = 0
        # print('a_u:', a_u, 'a_v:', a_v, 'a_w:', a_w)
        return a_u, a_v, a_w

    '''计算后向a_u, a_v, a_w'''
    def calculateA_b(self):
        # uw_aver = -math.pow(self.u_star, 2)
        # 20230315
        uw_aver = -math.pow(self.u_star, 2) * math.pow(1 - self.Z, 2 / 3)
        vw_aver = 0
        uv_aver = 0
        # pd_uw_aver_to_z = 0
        # 20230315
        pd_uw_aver_to_z = 2 / (3 * self.Z_i) * math.pow(self.u_star, 2) * math.pow(1 - self.Z, -1 / 3)
        pd_vw_aver_to_z = 0
        pd_sigma_u_s_to_z, pd_sigma_v_s_to_z, pd_sigma_w_s_to_z = self.calculatePd()
        b = math.pow(self.C0*self.epsilon, 1/2)

        '''6月7日修改'''
        a_u = -math.pow(b, 2) / (2 * self.sigma_s) * (self.u * (self.sigma_v_s * self.sigma_w_s - math.pow(vw_aver, 2))
                                                      + self.v * uw_aver * vw_aver - self.w * self.sigma_v_s * uw_aver) + (-1)*(1 / 2 * pd_uw_aver_to_z
        +self.u * self.w / (2 * self.sigma_s) * (pd_sigma_u_s_to_z * (
                self.sigma_v_s * self.sigma_w_s - math.pow(vw_aver, 2)) - pd_uw_aver_to_z * self.sigma_v_s * uw_aver)
        +self.v * self.w / (2 * self.sigma_s) * (
                    pd_sigma_u_s_to_z * uw_aver * vw_aver - pd_uw_aver_to_z * self.sigma_u_s * vw_aver)
        +math.pow(self.w, 2) / (2 * self.sigma_s) * (
                pd_uw_aver_to_z * self.sigma_u_s * self.sigma_v_s - pd_sigma_u_s_to_z * self.sigma_v_s * uw_aver))   #SD removed (-1) #SDSD restored -1
        # '''
        a_v = -math.pow(b, 2) / (2 * self.sigma_s) * (self.u * uw_aver * vw_aver + self.v * (
                self.sigma_u_s * self.sigma_w_s - math.pow(uw_aver, 2)) - self.w * self.sigma_u_s * vw_aver) + (-1)*(1 / 2 * pd_vw_aver_to_z
        +self.u * self.w / (2 * self.sigma_s) * (
                    pd_sigma_v_s_to_z * uw_aver * vw_aver - pd_vw_aver_to_z * self.sigma_v_s * uw_aver)
        +self.v * self.w / (2 * self.sigma_s) * (pd_sigma_v_s_to_z * (
                self.sigma_u_s * self.sigma_w_s - math.pow(uw_aver, 2) - pd_vw_aver_to_z * self.sigma_u_s * vw_aver))
        +math.pow(self.w, 2) / (2 * self.sigma_s) * (
                pd_vw_aver_to_z * self.sigma_u_s * self.sigma_v_s - pd_sigma_v_s_to_z * self.sigma_u_s * vw_aver)) # SD removed (-1) #SDSD restored -1
        # '''
        # a_v = -self.C0*self.epsilon/(2*self.sigma_v_s)*self.v #20240222
        # '''
        a_w = math.pow(b, 2) / (2 * self.sigma_s) * (
                self.u * self.sigma_v_s * uw_aver + self.v * self.sigma_u_s * vw_aver - self.w * self.sigma_u_s * self.sigma_v_s) + (-1)*(1 / 2 * pd_sigma_w_s_to_z
        +self.u * self.w / (2 * self.sigma_s) * (pd_uw_aver_to_z * (self.sigma_v_s * self.sigma_w_s - math.pow(vw_aver,
                        2)) + pd_vw_aver_to_z * uw_aver * vw_aver - pd_sigma_w_s_to_z * self.sigma_v_s * uw_aver)
        +self.v * self.w / (2 * self.sigma_s) * (pd_uw_aver_to_z * uw_aver * vw_aver + pd_vw_aver_to_z * (
                self.sigma_u_s * self.sigma_w_s - math.pow(uw_aver, 2) - pd_sigma_w_s_to_z * self.sigma_u_s * vw_aver))
        +math.pow(self.w, 2) / (2 * self.sigma_s) * (
                pd_sigma_w_s_to_z * self.sigma_u_s * self.sigma_v_s - pd_uw_aver_to_z * self.sigma_v_s * uw_aver - pd_vw_aver_to_z * self.sigma_u_s * vw_aver))  #SD  removed (-1) #SDSD restored -1
        # '''
        # a_w = -self.C0*self.epsilon/(2*self.sigma_w_s)*self.w+1/2*pd_sigma_w_s_to_z*(1+math.pow(self.w/self.sigma_w, 2)) # 20240102v5
        # a_w = -self.C0*self.epsilon/(2*self.sigma_w_s)*self.w-1/2*pd_sigma_w_s_to_z*(1+math.pow(self.w/self.sigma_w, 2)) # 20240102v6
        # a_u, a_v = 0, 0 # 20240125
        # a_u = 0
        # print('a_u:', a_u, 'a_v:', a_v, 'a_w:', a_w)
        return a_u, a_v, a_w


    '''计算du，dv,dw '''
    def calculateDeltaVelocity(self):
        a_u, a_v, a_w = self.calculateA()
        '''dξu，dξw，dξv基于高斯分布生成'''
        delta_ksi_u, delta_ksi_w, delta_ksi_v = np.random.normal(loc=0.0, scale=math.pow(self.dt, 1/2), size=3)
        delta_u = a_u*self.dt+math.pow(self.C0*self.epsilon, 1/2)*delta_ksi_u
        delta_v = a_v*self.dt+math.pow(self.C0*self.epsilon, 1/2)*delta_ksi_v
        delta_w = a_w*self.dt+math.pow(self.C0*self.epsilon, 1/2)*delta_ksi_w
        return a_u, a_v, a_w, delta_u, delta_v, delta_w

    '''计算后向du，dv,dw '''
    def calculateDeltaVelocity_b(self):
        a_u, a_v, a_w = self.calculateA_b()
        '''dξu，dξw，dξv基于高斯分布生成'''
        delta_ksi_u, delta_ksi_w, delta_ksi_v = np.random.normal(loc=0.0, scale=math.pow(self.dt, 1/2), size=3)
        delta_u = a_u * self.dt + math.pow(self.C0 * self.epsilon, 1 / 2) * delta_ksi_u  # SD multiply a by (-1) #SDSD removed -1
        delta_v = a_v * self.dt + math.pow(self.C0 * self.epsilon, 1 / 2) * delta_ksi_v  # SD multiply a by (-1) #SDSD removed -1
        delta_w = a_w * self.dt + math.pow(self.C0 * self.epsilon, 1 / 2) * delta_ksi_w  # SD multiply a by (-1) #SDSD removed -1
        return a_u, a_v, a_w, delta_u, delta_v, delta_w


    '''更新粒子脉动速度u,v,w'''
    def updateUVW(self):
        a_u, a_v, a_w, delta_u, delta_v, delta_w = self.calculateDeltaVelocity()
        u = self.u+delta_u
        v = self.v+delta_v
        w = self.w+delta_w
        return a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w

    '''更新后向粒子脉动u，v，w'''
    def updateUVW_b(self):
        a_u, a_v, a_w, delta_u, delta_v, delta_w = self.calculateDeltaVelocity_b()
        u = self.u + delta_u
        v = self.v + delta_v
        w = self.w + delta_w
        return a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w

    '''计算dx，dy，dz'''
    def calculateDeltaDis(self):
        a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w = self.updateUVW()
        delta_x = (self.U_aver_update+u) * self.dt
        # delta_x = self.U_aver_update*self.dt  # 20240102
        # delta_y = (self.V_aver_update+v) * self.dt
        delta_y = v*self.dt # 20240222
        # delta_z = (self.W_aver_update+w) * self.dt
        delta_z = w*self.dt # 20240102
        return a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w, delta_x, delta_y, delta_z

    '''计算后向dx，dy，dz'''
    def calculateDeltaDis_b(self):
        a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w = self.updateUVW_b()
        delta_x = -(self.U_aver_update + u) * self.dt  # SD changed  +u to - u #SDSD changed back to +
        # delta_x = -self.U_aver_update*self.dt  # 20240102
        # delta_y = -(self.V_aver_update + v) * self.dt   # SD changed  +v to -v #SDSD changed back to +
        delta_y = -v*self.dt
        # delta_z = -(self.W_aver_update + w) * self.dt   # SD changed  +w to -w #SDSD changed back to +
        delta_z = -w*self.dt
        return a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w, delta_x, delta_y, delta_z

    '''更新x，y，z'''
    def updateXYZ(self):
        a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w, delta_x, delta_y, delta_z = self.calculateDeltaDis()
        x_wc = self.x_wc + delta_x
        y_wc = self.y_wc + delta_y
        z = self.z + delta_z
        return a_u, a_v, a_w, u, v, w, x_wc, y_wc, z

    '''更新后向x，y，z'''
    def updateXYZ_b(self):
        a_u, a_v, a_w, delta_u, delta_v, delta_w, u, v, w, delta_x, delta_y, delta_z = self.calculateDeltaDis_b()
        x_wc = self.x_wc + delta_x
        y_wc = self.y_wc + delta_y
        z = self.z + delta_z
        return a_u, a_v, a_w, u, v, w, x_wc, y_wc, z
