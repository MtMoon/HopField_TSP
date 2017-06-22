# coding=utf-8
from __future__ import division
import numpy as np
import random
import copy
import time
import math
import pylab as plt
'''
hopfield
'''


class Hopfield(object):

    def __init__(self, datafile):


        self.A = 500  #  row
        self.B = 500 # colunm
        self.C = 1000 # total n
        self.D = 500  # distance

        self.__load_data(datafile)

        self.gamma = 0.02
        self.max_time_step = 1000
        self.max_run_num = 200
        
        # RK
        self.RK_delta_t = 0.005
        self.tau = 1.0

    def __load_data(self, file):
        fin = open(file, 'r')
        lines = fin.readlines()
        self.N = len(lines)  # city number

        self.city_list = []
        for line in lines:
            line = line.strip()
            vals = line.split(" ")
            self.city_list.append((float(vals[0]), float(vals[1])))

        # build distance dictionary
        self.dis_dic = np.zeros([self.N, self.N], dtype=np.float32)
        for i in xrange(0, self.N):
            for j in xrange(i+1, self.N):
                c1 = self.city_list[i]
                c2 = self.city_list[j]
                dis = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                self.dis_dic[i, j] = dis
                self.dis_dic[j, i] = dis


    def dy_dt(self, y_ai, a, i):
        ans = -y_ai / self.tau 
        tmp = np.sum(self.X[a, :]) - self.X[a, i]
        tmp *= self.A
        ans -= tmp

        tmp = np.sum(self.X[:, i]) - self.X[a, i]
        tmp *= self.B
        ans -= tmp

        tmp = np.sum(self.X) - self.N
        tmp *= self.C
        ans -= tmp

        tmp = 0.0
        for b in xrange(0, self.N):
            if b == a:
                continue

            x1 = self.X[b, (i+1)%self.N]
            x0 = self.X[b, (i-1)%self.N]
            tmp += self.dis_dic[a, b] * (x0+x1)

        tmp *= self.D
        ans -= tmp

        return ans

    # 2-order RK algorithm to calculate y
    def RK2(self, a, i):
        #print ("begin")
        y_ai = self.Y[a, i]
        loss = 0.0
        last_loss = 0.0
        for ite in range(0, self.RK_max_iter):
            #print (y_ai)
            k1 = self.dy_dt(y_ai, a, i)
            k2 = self.dy_dt(y_ai + self.RK_delta_t*k1, a, i)
            #print (k1, k2)
            new_y_ai = y_ai + 0.5*self.RK_delta_t*(k1+k2)

            loss = np.abs(new_y_ai-y_ai)
            y_ai = new_y_ai
            if np.abs(loss-last_loss) < self.RK_esp:
                break
            last_loss = loss
        #print (ite)
        #print ("________________")
        return y_ai


    def calculate_energy(self):
        J = 0.0
        for a in xrange(0, self.N):
            for b in xrange(0, self.N):
                if a == b:
                    continue
                for i in xrange(0, self.N):
                    x1 = self.X[b,(i+1)%self.N]
                    x0 = self.X[b,(i-1)%self.N]
                    J += self.dis_dic[a, b] * self.X[a,i] * (x1+x0) 

        J1 = 0.0
        for a in xrange(0, self.N):
            for i in xrange(0, self.N):
                for  j in xrange(0, self.N):
                    if j == i:
                        continue
                    J1 += self.X[a,i]*self.X[a,j]

        J2 = 0.0
        for a in xrange(0, self.N):
            for i in xrange(0, self.N):
                for  b in xrange(0, self.N):
                    if b == a:
                        continue
                    J1 += self.X[a,i]*self.X[b,i]

        J3 = (np.sum(self.X)-self.N)**2

        #print ("J:%.2f J1:%.2f J2:%.2f J3:%.2f E:%.2f" % (J, J1, J2, J3, self.A*J1+self.B*J2+self.C*J3+self.D*J))
        #return  self.A*J1+self.B*J2+self.C*J3+self.D*J
        return  [self.A*J1, self.B*J2, self.C*J3, self.D*J]


    def checkans(self, showE=False):
        cities = []
        for i in xrange(0, self.N):
            idx = np.argmax(self.X[:, i])
            cities.append(idx)

        #print ("epoch %d" % (step))
        #print (cities)
        dis = 0.0
        cities.append(cities[0])
        for i in xrange(0, len(cities)-1):
            dis += self.dis_dic[cities[i], cities[i+1]]

        if showE:
            energy = self.calculate_energy()
        else:
            energy = -1

        flag = True
        for i in xrange(0, self.N):
            if not i in cities:
                flag = False
                break

        if flag:
            return True, dis, cities, energy
        else:
            return False, -1, [], energy

    def process(self):

        '''
        initial = range(0, self.N)
 
        random.shuffle(initial)
        for i in xrange(0, self.N):
            self.X[i, initial[i]] = 1.0
        '''
        legal_count = 0
        best_count = 0
        fout = open("anslog.txt", 'w')

        for itera in xrange(0, self.max_run_num):
            print ("run %d time(s)..." % (itera))
            # build x matrix
            #self.X = np.zeros([self.N, self.N], dtype=np.float32)
            best_dis, best_path, best_step = self.run()
            #print (np.round(self.X, 2))
            if best_step >= 0:
                legal_count += 1
                if np.abs(best_dis-2.6907) < 0.0001:
                    best_count += 1
                log_str = str((itera, best_dis, " ".join([str(c) for c in best_path]), best_step))
                print (log_str)
                fout.write(str(log_str) + "\n")
                fout.flush()

        print (legal_count)
        fout.write("\n")
        fout.write("legal_count: " + str(legal_count) + "\n")
        fout.write("best_count: " + str(best_count) + "\n")
        fout.close()

    def sigmoid(self, Y):

        ans = 1 / (1.0+np.exp(-2/self.gamma*Y))
        #final_ans = ans
        #print (ans)
        
        upper = ans > 0.8
        upper = upper.astype(np.float32)

        lower = ans < 0.2
        lower = 1-lower.astype(np.float32)

        final_ans = ans * lower * (1-upper) + upper

        return final_ans


    def run(self):
        best_dis = 1000
        best_path = 1000
        best_step = -1

        self.Y = np.ones([self.N, self.N], dtype=np.float32) * np.log(self.N-1.0) * (-self.gamma/2.0)
        self.Y = self.Y * (np.random.rand(self.N, self.N)*0.2+0.9)
        self.X = self.sigmoid(self.Y)

        J1 = []
        J2 = []
        J3 = []
        J = []
        total = []


        for step in xrange(0, self.max_time_step):
            DY = np.zeros([self.N, self.N], dtype=np.float32)
            for a in xrange(0, self.N):
                for i in xrange(0, self.N):
                    
                    y_ai = self.Y[a, i]
                    #DY[a, i] = self.dy_dt(y_ai, a, i)
                    k1 = self.dy_dt(y_ai, a, i)
                    k2 = self.dy_dt(y_ai + self.RK_delta_t*k1, a, i)
                    DY[a,i] = 0.5*self.RK_delta_t*(k1+k2)
       
            self.Y = self.Y + self.RK_delta_t*DY
            #self.X = (1.0+np.tanh(self.Y/self.gamma)) / 2.0
            self.X = self.sigmoid(self.Y)
            flag, dis, path, energy = self.checkans(True)
            J1.append(energy[0])
            J2.append(energy[1])
            J3.append(energy[2])
            J.append(energy[3])
            total.append(np.sum(energy))
            #print ("energy: %f" % (energy))
            if flag and dis < best_dis:
                best_dis = dis
                best_path = path
                best_step = step
        self.draw(J1, J2, J3, J, total)
        return best_dis, best_path, best_step

    def draw(self, J1, J2, J3, J, total):
  
        x = xrange(0, len(J1))
        p1 = plt.subplot(221)
        p1.plot(x, np.array(J1)/100000, 'r-')
        p1.set_title("J1")
        plt.xlabel("iterations")  
        plt.ylabel("Energy / 1e5")  

        p2 = plt.subplot(222)
        p2.plot(x, J2, 'b-')
        p2.set_title("J2")
        plt.xlabel("iterations")  
        plt.ylabel("Energy")  

        p3 = plt.subplot(223)
        p3.plot(x, np.array(J3)/100000, 'g-')
        p3.set_title("J3")
        plt.xlabel("iterations")  
        plt.ylabel("Energy / 1e5")  

        p4 = plt.subplot(224)
        p4.plot(x, np.array(J)/100000, 'm-')
        p4.set_title("J")
        plt.suptitle('Energy Curve')
        plt.xlabel("iterations")  
        plt.ylabel("Energy / 1e5")  
        
        plt.show()




def main():
    hopfield = Hopfield("TSP10_norm.txt")
    hopfield.run()
    #hopfield.process()


if __name__ == "__main__":
    main()
