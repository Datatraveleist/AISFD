import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegression(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_prob=0.2, output_size=1):
        super(MLPRegression, self).__init__()
        layers = []
        prev_size = input_size

        for i, layer_size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, layer_size))  
            layers.append(nn.ReLU())  

            if i < 1:
                layers.append(nn.Dropout(dropout_prob))  
            
            prev_size = layer_size  
        

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

        # print("Network Layers:")
        # for layer in layers:
        #     print(layer)
            
    def forward(self, x):
        out = self.model(x)
        out = out.squeeze(1)
        #return self.model(x)[:, 0] 
        return out

class ModelC_T(nn.Module):
    def __init__(self,layer_sizes):
        super(ModelC_T, self).__init__()
        self.layer_sizes = layer_sizes
        self.model = MLPRegression(input_size=256*2, layer_sizes=self.layer_sizes, dropout_prob=0.3, output_size=1)

    def forward(self, x):
        return self.model(x)

class ModelCstar(nn.Module):
    def __init__(self,layer_sizes):
        super(ModelCstar, self).__init__()
        self.layer_sizes = layer_sizes
        self.model = MLPRegression(input_size=256*2, layer_sizes=self.layer_sizes, dropout_prob=0.3, output_size=1)

    def forward(self, x):
        return self.model(x)

class ModelIsp(nn.Module):
    def __init__(self,layer_sizes):
        super(ModelIsp, self).__init__()
        self.layer_sizes = layer_sizes
        self.model = MLPRegression(input_size=256*2, layer_sizes=self.layer_sizes, dropout_prob=0.3, output_size=1)

    def forward(self, x):
        return self.model(x)
    
class Features_Formulation(nn.Module):
    def __init__(self):
        super(Features_Formulation, self).__init__()
        self.hidden1 = nn.Linear(in_features = 10, out_features=128, bias=True)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden3 = nn.Linear(256, 256)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        output = F.relu(self.hidden3(x))
        return output

class Features_EMS(nn.Module):
    def __init__(self):
        super(Features_EMS, self).__init__()
        self.hidden1 = nn.Linear(in_features = 65, out_features=128, bias=True)
        self.hidden2 = nn.Linear(128, 256)
        self.hidden3 = nn.Linear(256, 256)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        output = F.relu(self.hidden3(x))
        return output

# class features_Estate(nn.Module):
#     def __init__(self):
#         super(features_Estate, self).__init__()
#         self.hidden1 = nn.Linear(in_features = 78, out_features=128, bias=True)
#         self.hidden2 = nn.Linear(128, 256)
#     def forward(self, x):
#         x = F.relu(self.hidden1(x))
#         output = F.relu(self.hidden2(x))
#         return output[:, 0]

class Model_isp_all(nn.Module):
    def __init__(self,layer_sizes):
        super(Model_isp_all, self).__init__()
        self.layer_sizes = layer_sizes
        self.features_formulation = Features_Formulation()
        self.features_ems = Features_EMS()
        self.modelIsp = ModelIsp(self.layer_sizes)
    def forward(self, X):
        #if not torch.is_tensor(X):
         #   X = torch.Tensor(X)
        formulation_x, ems_x = X[:, 0:10], X[:, 10:-1]
        formulation_out = self.features_formulation(formulation_x)
        ems_out = self.features_ems(ems_x)
        # print(formulation_out,'formulation_out')
        # print(ems_out,'ems_out')
        isp_input = torch.cat((formulation_out, ems_out), 1)    
        out = self.modelIsp(isp_input)
        return out 
        
class Model_cstar_all(nn.Module):
    def __init__(self,layer_sizes):
        super(Model_cstar_all, self).__init__()
        self.layer_sizes = layer_sizes
        self.features_formulation = Features_Formulation()
        self.features_ems = Features_EMS()
        self.modelCstar = ModelCstar(self.layer_sizes)
    def forward(self, X):
        formulation_x, ems_x = X[:, 0:10], X[:, 10:-1]
        formulation_out = self.features_formulation(formulation_x)
        ems_out = self.features_ems(ems_x)
        Cstar_input = torch.cat((formulation_out, ems_out), 1)    
        out = self.modelCstar(Cstar_input)
        return out      

class Model_c_t_all(nn.Module):
    def __init__(self,layer_sizes):
        super(Model_c_t_all, self).__init__()
        self.layer_sizes = layer_sizes
        self.features_formulation = Features_Formulation()
        self.features_ems = Features_EMS()
        self.modelC_T = ModelC_T(self.layer_sizes)
    def forward(self, X):
        formulation_x, ems_x = X[:, 0:10], X[:, 10:-1]
        formulation_out = self.features_formulation(formulation_x)
        ems_out = self.features_ems(ems_x)
        C_T_input = torch.cat((formulation_out, ems_out), 1)    
        out = self.modelC_T(C_T_input)
        return out   