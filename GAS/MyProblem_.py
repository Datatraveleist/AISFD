import numpy as np
import geatpy
import geatpy as ea
np.random.seed(42)
from predict import input_geatpy_isp
data = []
# print(geatpy.Problem)
class MyProblem(ea.Problem): # 
    def __init__(self,smiles,n1_max,n1_min,n2_max,n2_min,n3_max,n3_min,n4_max,n4_min):
    # def __init__(self,smiles):
        name = 'ZDT1' # 
        M = 1 # 
        maxormins = [-1] * M # 
        Dim = 4 # 
        varTypes = [0] * Dim # 
        lb = [0] * Dim # 
        ub = [99] * Dim # 
        lbin = [1] * Dim # 
        ubin = [1] * Dim # 
        self.smiles = smiles

        self.n1_max,self.n1_min = n1_max,n1_min
        self.n2_max,self.n2_min = n2_max,n2_min
        self.n3_max,self.n3_min = n3_max,n3_min
        self.n4_max,self.n4_min = n4_max,n4_min
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    
    
    def aimFunc(self, pop): 
        Vars = pop.Phen 
        # print(Vars.shape)
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
 
        # x1 = np.clip(x1, self.n1_min, self.n1_max)
        # x2 = np.clip(x2, self.n2_min, self.n2_max)
        # x3 = np.clip(x3, self.n3_min, self.n3_max)
        # x4 = np.clip(x4, self.n4_min, self.n4_max)
        # print(input_geatpy_isp(x1,x2,x3,x4))
        pop.ObjV = input_geatpy_isp(x1,x2,x3,x4,[self.smiles]) 
        # pop.CV = np.hstack([np.abs(x1+x2+x3+x4-100),np.abs(x3-12),np.abs(x1-18)])
        pop.CV = np.hstack([np.abs(x1+x2+x3+x4-100),self.n2_min-x2,x2-self.n2_max,
                            self.n3_min-x3,x3-self.n3_max,
                            self.n1_min-x1,x1-self.n1_max])

# input_geatpy_isp(1,2,3,4,['CC'])
