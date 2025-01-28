#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/02/15
# @Author  :
# @Site    :
# @File    : CalculateStatistics.py
# @Software: PyCharm

import math

'''
面源计算；对指标L,𝜙_M,𝜕𝑈/𝜕𝑧、 𝜙_1,𝑈(𝑧), ϵ进行计算
'''

class CalculateBasicIndex(object):
    def __init__(self, u_star, w_star, z, Z_i, z_0,  L, scenario, k=0.4, C0=3.0):
        self.u_star = u_star
        self.w_star = w_star
        self.z = z
        self.Z_i = Z_i
        self.Z = self.z/self.Z_i
        self.z_0 = z_0
        self.L = L
        '''在下述情形中，需要对L进行重新计算'''
        if self.L < 0 and self.w_star > 0:
            self.L = self.calculateL()
        '''在下述情形中，需要对z进行重新计算'''
        self.scenario = scenario
        self.k = k
        self.C0 = C0

    # 计算L
    def calculateL(self):
        L = -math.pow(self.u_star / self.w_star, 3) * self.Z_i / 0.4
        return L


    # 计算ϕ_M
    # def calculatePhi_M(self):
    #     if self.z/self.L > 0:
    #         phi_M = 1 + 5 * (self.z / self.L)
    #     elif self.z/self.L < 0:
    #         phi_M = math.pow(1 - 16 * self.z / self.L, -1 / 4)
    #     else:
    #         print('wrong z (z=0) input')
    #         print('from /CalculateBasicIndex/ class /calculatePhi_M/ function')
    #         exit()
    #         return
    #     return phi_M


    def calculatePhi_M(self):
        if self.z/self.L > 0:
            phi_M = 1 + 5 * (self.z / self.L)
        elif self.z/self.L < 0:
            phi_M = math.pow(1 - 16 * self.z / self.L, -1 / 4)
        else:
            print('z, L:', self.z, self.L)
            print('wrong input')
            return
        return phi_M

    # 计算风速梯度∂U/∂z，即pd_U_aver_to_z
    def calculateVerGradofWD(self):
        phi_M = self.calculatePhi_M()
        # try:
        #     phi_M = self.calculatePhi_M()
        # except Exception as e:
        #     print('z, L:', self.z, self.L)
        #     exit()
        pd_U_aver_to_z = self.u_star / (self.k * self.z) * phi_M
        return pd_U_aver_to_z


    # 计算风速U（z），计算的为某坐标气象条件下某高度z下的风速，是否可以理解为矫正
    # 需要将其作为粒子在某高度处的速度
    def calculateWD(self):
        if self.L > 0:
            # print('self.z:', self.z, 'self.z_0:', self.z_0)
            # print('self.u_star', self.u_star, 'self.z/self.z_0:', self.z / self.z_0)
            U_z = self.u_star / self.k * (math.log(self.z / self.z_0) + 5 * self.z / self.L)
        elif self.L < 0:
            phi_M = self.calculatePhi_M()
            x = 1/phi_M
            phi_1 = 2 * math.log((1 + x) / 2) + math.log((1 + math.pow(x, 2)) / 2) - 2 * math.atan(
                x) + math.pi / 2  # 这里的math.atan(x)返回的为弧度
            # print('self.z:', self.z)
            U_z = self.u_star / self.k * (math.log(self.z / self.z_0) - phi_1)
        else:
            print('wrong Monin-Obukhov length (L=0) input')
            print('from /CalculateBasicIndex/ class /calculateWD/ function')
            # exit()
            return
        return U_z

    '''计算各分量方差'''
    def calculateSigmaSquare(self):
        if self.L > 0: # 非对流
            sigma_w_s = 1.7 * math.pow(self.u_star, 2) * math.pow(1 - self.Z, 3 / 2)
            sigma_u_s = 4.0 * math.pow(self.u_star, 2) * math.pow(1 - self.Z, 3 / 2)
            sigma_v_s = 5.0 * math.pow(self.u_star, 2) * math.pow(1 - self.Z, 3 / 2)
        else: # L < 0
            if -self.Z_i / self.L < 1000:  # 非对流
                sigma_w_s = 1.7 * math.pow(self.u_star, 2) * math.pow(1 - self.z/self.L, 2 / 3)
                sigma_v_s = 0.35*math.pow(self.w_star, 2)+2*math.pow(self.u_star, 2)*(1-0.5*self.Z)
                sigma_u_s = sigma_v_s
            else: # Z_i/L>=10 对流
                b1, b2, b3 = 0.0020, 1.2, 0.333
                sigma_w_s = math.pow(self.w_star, 2) * math.pow(abs(b1 + b2 * self.Z * (1 - self.Z) * (1 - b3 * self.Z)), 2 / 3)  # 20230107
                sigma_v_s = 0.35 * math.pow(self.w_star, 2) + 2 * math.pow(self.u_star, 2) * (1 - 0.5 * self.Z)  # 面源
                # sigma_v_s = 3.6 * math.pow(self.u_star, 2) + 0.35 * math.pow(self.w_star, 2)
                sigma_u_s = sigma_v_s
        return sigma_u_s, sigma_v_s, sigma_w_s


    # Dissipation rate of turbulent kinetic energy
    # 计算湍流动能耗散率
    def calculateEpsilon(self, sigma_w_s):
        if self.scenario == 'CBL':
            b1, b2, b3, b4 = 0.0020, 1.2, 0.333, 0.72
            w_square_aver = math.pow(self.w_star, 2) * math.pow((b1 + b2 * self.Z * (1 - self.Z) * (1 - b3 * self.Z)), 2 / 3)
            epsilon = 4*math.pow(w_square_aver, 3/2)*math.exp(self.Z)/(self.C0*self.z*math.pow(1-6*self.z/self.L, 1/4))
            # try:
            #     w_square_aver = math.pow(self.w_star, 2) * math.pow(
            #         (b1 + b2 * self.Z * (1 - self.Z) * (1 - b3 * self.Z)), 2 / 3)
            #     epsilon = 4*math.pow(w_square_aver, 3/2)*math.exp(self.Z)/(self.C0*self.z*math.pow(1-6*self.z/self.L, 1/4))
            # except Exception as e:
            #     w_square_aver = math.pow(self.w_star, 2) * math.pow(
            #         (b1 + b2 * self.Z * (1 - self.Z) * (1 - b3 * self.Z)), 2 / 3)
            #     print('w_square_aver:', w_square_aver, '1-6*self.z/self.L:', 1-6*self.z/self.L)
            #     exit()
        else:
            # sigma_u_s, sigma_v_s, sigma_w_s = self.calculateSigmaSquare()
            sigma_w = math.pow(sigma_w_s, 1/2)
            if self.L > 0:
                T_L = self.z/(2*sigma_w)*math.pow(1+5*self.z/self.L, -1)
            elif self.L == 0:
                print('wrong Monin-Obukhov length (L=0) input')
                print('from /CalculateBasicIndex/ class /calculateEpsilon/ function')
                # exit()
                return
            else:  # L<0
                T_L = self.z / (2 * sigma_w) * math.pow(1 - 6 * self.z / self.L, 1/4)
            epsilon = 2 * sigma_w_s / (self.C0 * T_L)
        # phi_M = self.calculatePhi_M()
        # epsilon = math.pow(self.u_star, 3) / (self.k * self.z) * phi_M + math.pow(self.u_star, 3) / (self.k * self.L) * (1 - self.z / self.z_i)
        return epsilon