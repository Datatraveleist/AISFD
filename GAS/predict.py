import pandas as pd
import numpy as np
import pickle
np.random.seed(42)
from rdkit import Chem
from descriptors_generator import generator
from rdkit.Chem import AllChem
from ase.io import read
from CEA_Wrap import Fuel, Oxidizer, RocketProblem, DataCollector
from rdkit.Chem.rdmolfiles import MolToXYZFile
from MLP_frame import Model_isp_all,Model_c_t_all,Model_cstar_all
import torch
import random
def cal_isp(x1,x2,x3,x4,Hf,smiles):  
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    MolToXYZFile(mol,'mol.xyz')
    xyz = read('mol.xyz')
    atom_types = xyz.get_chemical_symbols()
    atom_types_ls = set(atom_types)
    formula_ls = []
    for atom_type in atom_types_ls:
        formula_ls.append(atom_type)
        count = 0
        for a in atom_types:
            if a == atom_type:
                count += 1
        formula_ls.append(str(count))
        #print(formula_ls)
        formula_str =' '.join(formula_ls)
    # print(ems,Hf)
    AL = Fuel("AL(cr)", wt_percent=x1)  # (cr) for "crystalline" or condensed phase
    EMs = Oxidizer("EMs", wt_percent=x2, chemical_composition = formula_str, hf = Hf)
    #EMs.check_against_thermo_inp = False
    # MMH = Oxidizer("MDH", wt_percent=27, chemical_composition = 'C 4 H 1 O 10 N 7', hf = 54.14) # This was added at Purdue so doesn't include (cr) in the name
    HTPB = Fuel("HTPB", wt_percent=x3)
    #HTPB = Fuel("GAP", wt_percent=cpd[2], chemical_composition = 'C 3 H 5 N 3 O 1', hf = 141.939)
    AP = Oxidizer("NH4CLO4(I)", wt_percent=x4)# ammonium perchlorate (form I, specified at room temperature)
    #OLD_EMS = Oxidizer("RDX", wt_percent=cpd[4], chemical_composition = cpd2_f, hf = eof2)
    m_list = [AL, EMs, HTPB, AP] # for convenience so I can pass it into all problems
    problem = RocketProblem(materials=m_list, pressure = 69.8, pressure_units = 'bar', pip=69.8)
    problem.set_absolute_o_f() # have it calculate o_f for us from material percentage
    collector = DataCollector("c_t", "isp", "cstar") # Show chamber temperature and isp dependence on % aluminum
    problem.set_absolute_o_f() # change our o/f to reflect situation
    collector.add_data(problem.run())
    result0 = collector.c_t
    result1 = collector.isp
    result2 = collector.cstar               
    return result0 ,result1 , result2

def data_type_delete(data_type):
    data = pd.DataFrame(data_type)
    X = data[['Al','EMs','HTPB','NH4CLO4','C_mol','H_mol','O_mol','N_mol','Al_mol','Cl_mol','wt_H','nHbondA', 'nHbondD', 'nNH2', 'nAHC', 'nACC', 'nHC', 'nRbond', 'nR', 'nNNO2', 'nONO2', 'nNO2', 'nC(NO2)3', 'nC(NO2)2', 'nC(NO2)', 'MinPartialCharge', 'MaxPartialCharge', 'MOLvolume', 'nH', 'nC', 'nN', 'nO', 'PBF', 'TPSA', 'ob', 'total energy', 'molecular weight', 'PMI3', 'nOCH3', 'nCH3', 'Eccentricity', 'PMI2', 'PMI1', 'NPR1', 'NPR2', 'ESTATE_0', 'ESTATE_1', 'ESTATE_2', 'ESTATE_3', 'ESTATE_4', 'ESTATE_5', 'ESTATE_6', 'ESTATE_7', 'ESTATE_8', 'ESTATE_9', 'ESTATE_10', 'ESTATE_11', 'ESTATE_12', 'ESTATE_13', 'ESTATE_14', 'ESTATE_15', 'ESTATE_16', 'ESTATE_17', 'ESTATE_18', 'ESTATE_19', 'ESTATE_20', 'ESTATE_21', 'ESTATE_22', 'ESTATE_23', 'ESTATE_24', 'ESTATE_25', 'ESTATE_26', 'ESTATE_27', 'ESTATE_28', 'ESTATE_29', 'ESTATE_30', 'ESTATE_31', 'ESTATE_32', 'ESTATE_33', 'ESTATE_34', 'ESTATE_35', 'ESTATE_36', 'ESTATE_37', 'ESTATE_38', 'ESTATE_39', 'ESTATE_40', 'ESTATE_41', 'ESTATE_42', 'ESTATE_43', 'ESTATE_44', 'ESTATE_45', 'ESTATE_46', 'ESTATE_47', 'ESTATE_48', 'ESTATE_49', 'ESTATE_50', 'ESTATE_51', 'ESTATE_52', 'ESTATE_53', 'ESTATE_54', 'ESTATE_55', 'ESTATE_56', 'ESTATE_57', 'ESTATE_58', 'ESTATE_59', 'ESTATE_60', 'ESTATE_61', 'ESTATE_62', 'ESTATE_63', 'ESTATE_64', 'ESTATE_65', 'ESTATE_66', 'ESTATE_67', 'ESTATE_68',
                           'ESTATE_69']] 
    #MOLvolume,PMI2,PMI1
    unused_isp = ['nC(NO2)3','nC(NO2)2','nACC','nONO2','nOCH3','Eccentricity','nHC','total energy','PMI1','PMI2','PBF','PMI3','NPR1','NPR2','MOLvolume']
    # unused_isp = ['nACC','nONO2','nOCH3','Eccentricity','nHC','total energy','PMI1','PMI2','PBF','PMI3','NPR1','NPR2']
    # unused_cstar = ['Eccentricity','nHC','total energy','PMI1','PMI2','PBF','PMI3','NPR1','NPR2','ESTATE_43','ESTATE_12','ESTATE_19','ESTATE_17','ESTATE_4','nONO2','ESTATE_54','ESTATE_24','nC(NO2)3']
    # unused_c_t = ['Eccentricity','nHC','total energy','PMI1','PMI2','ESTATE_54','ESTATE_43','ESTATE_27','ESTATE_9','ESTATE_8','ESTATE_5','ESTATE_35','ESTATE_23','ESTATE_14','ESTATE_19','ESTATE_6']
    # unused = ['Eccentricity','nHC','total energy','PMI1','PMI2','ESTATE_0','ESTATE_58','ESTATE_54','ESTATE_19','ESTATE_17','ESTATE_16','ESTATE_4','nC(NO2)3']
    all_zero = ['ESTATE_3', 'ESTATE_7', 'ESTATE_13', 'ESTATE_15', 'ESTATE_20',
           'ESTATE_26', 'ESTATE_31', 'ESTATE_32', 'ESTATE_33', 'ESTATE_34',
           'ESTATE_38', 'ESTATE_42', 'ESTATE_48', 'ESTATE_50', 'ESTATE_55',
           'ESTATE_61', 'ESTATE_66', 'ESTATE_67', 'ESTATE_68', 'ESTATE_69']

    unused_isp1 = ['ESTATE_36','ESTATE_16','ESTATE_51','ESTATE_65']
    X=X.drop(all_zero,axis=1)
    # unused = []
    X=X.drop(unused_isp,axis=1)
    X=X.drop(unused_isp1,axis=1)
    return X

def get_components(df_train):
    components = {
        'Al': df_train['Al'] / 100,
        'EMs': df_train['EMs'] / 100,
        'HTPB': df_train['HTPB'] / 100,
        'NH4CLO4': df_train['NH4CLO4'] / 100
    }
    element_composition = {
        'NH4CLO4': {'H': 4, 'O': 4, 'N': 1, 'Cl': 1},
        'EMs': {'C': df_train['nC'], 'H': df_train['nH'], 'O': df_train['nO'], 'N': df_train['nN']},
        'Al': {'Al': 1},
        'HTPB': {'C': 10, 'H': 15.40, 'O': 0.07}
    }
    molecular_weight = {
        'Al': 26.98,
        'EMs': df_train['molecular weight'],
        'HTPB': 86,
        'NH4CLO4': 117.49
    }
    return components, element_composition, molecular_weight, df_train['nH'] * 100 / df_train['molecular weight']

def calculate_imaginary_formula(components, element_composition, molecular_weight, total_mass=1000):
    total_moles = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'Al': 0, 'Cl': 0}  
    for component, percentage in components.items():
        component_mass = total_mass * percentage  
        for element, count in element_composition.get(component, {}).items():
            mol = component_mass / molecular_weight[component]
            element_mol = count * mol
            total_moles[element] += element_mol  

    return total_moles

def get_simplified_formula(df_train):
    components, element_composition, molecular_weight, wt_H = get_components(df_train)
    total_moles = calculate_imaginary_formula(components, element_composition, molecular_weight)
    df_train['C_mol'] = total_moles['C']
    df_train['H_mol'] = total_moles['H']
    df_train['O_mol'] = total_moles['O']
    df_train['N_mol'] = total_moles['N']
    df_train['Al_mol'] = total_moles['Al']
    df_train['Cl_mol'] = total_moles['Cl']
    df_train['wt_H'] = wt_H
    return df_train

def data_input(X1, X2, X3, X4, smiles):
    X_new_1 = np.array([X1]).reshape(-1, 1)
    X_new_2 = np.array([X2]).reshape(-1, 1)
    X_new_3 = np.array([X3]).reshape(-1, 1)
    X_new_4 = np.array([X4]).reshape(-1, 1)
    size = X_new_1.size

    X_d = np.hstack((X_new_1, X_new_2, X_new_3, X_new_4))

    descriptor_data = generator(smiles)
    repeated_matrix = np.tile(descriptor_data, (size, 1, 1))
    X_p = np.array([repeated_matrix]).reshape(size, -1)
    X_ALL = np.concatenate((X_p, X_d), axis=1)

    composite_descriptors = ['nHbondA', 'nHbondD', 'nNH2', 'nAHC', 'nACC', 'nHC', 'nRbond', 'nR', 'nNNO2', 
                             'nONO2', 'nNO2', 'nC(NO2)3', 'nC(NO2)2', 'nC(NO2)', 'MinPartialCharge', 'MaxPartialCharge', 
                             'MOLvolume', 'nH', 'nC', 'nN', 'nO', 'PBF', 'TPSA', 'ob', 'total energy', 'molecular weight', 
                             'PMI3', 'nOCH3', 'nCH3', 'Eccentricity', 'PMI2', 'PMI1', 'NPR1', 'NPR2', 'ESTATE_0', 
                             'ESTATE_1', 'ESTATE_2', 'ESTATE_3', 'ESTATE_4', 'ESTATE_5', 'ESTATE_6', 'ESTATE_7', 'ESTATE_8', 
                             'ESTATE_9', 'ESTATE_10', 'ESTATE_11', 'ESTATE_12', 'ESTATE_13', 'ESTATE_14', 'ESTATE_15', 
                             'ESTATE_16', 'ESTATE_17', 'ESTATE_18', 'ESTATE_19', 'ESTATE_20', 'ESTATE_21', 'ESTATE_22', 
                             'ESTATE_23', 'ESTATE_24', 'ESTATE_25', 'ESTATE_26', 'ESTATE_27', 'ESTATE_28', 'ESTATE_29', 
                             'ESTATE_30', 'ESTATE_31', 'ESTATE_32', 'ESTATE_33', 'ESTATE_34', 'ESTATE_35', 'ESTATE_36', 
                             'ESTATE_37', 'ESTATE_38', 'ESTATE_39', 'ESTATE_40', 'ESTATE_41', 'ESTATE_42', 'ESTATE_43', 
                             'ESTATE_44', 'ESTATE_45', 'ESTATE_46', 'ESTATE_47', 'ESTATE_48', 'ESTATE_49', 'ESTATE_50', 
                             'ESTATE_51', 'ESTATE_52', 'ESTATE_53', 'ESTATE_54', 'ESTATE_55', 'ESTATE_56', 'ESTATE_57', 
                             'ESTATE_58', 'ESTATE_59', 'ESTATE_60', 'ESTATE_61', 'ESTATE_62', 'ESTATE_63', 'ESTATE_64', 
                             'ESTATE_65', 'ESTATE_66', 'ESTATE_67', 'ESTATE_68', 'ESTATE_69', 'Al', 'EMs', 'HTPB', 'NH4CLO4']

    X_ALL = pd.DataFrame(X_ALL, columns=composite_descriptors)
    X_ALL = get_simplified_formula(X_ALL)
    
    X = data_type_delete(X_ALL)  # Assume `data_type()` is a predefined function
    with open('ss_X.pkl', 'rb') as ex:
        normalizing_x = pickle.load(ex)  
    X_input_Scaler = normalizing_x.transform(X)
    return np.array([X_input_Scaler]).reshape(size, -1)
def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def input_geatpy_isp(x1, x2, x3, x4,smiles):
    X_input_Scaler = data_input(x1, x2, x3, x4,smiles)
    X_input_Scaler = torch.Tensor(X_input_Scaler.astype(np.float32))  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_isp = Model_isp_all([832,704,1024,288,64,704,384,1024])
    model_isp.load_state_dict(torch.load('isp_best_network.pth', 
                                         map_location=device))  

    model_isp.eval()
    
    model_isp = model_isp.to(device)
    
    isp_train_p = model_isp(X_input_Scaler.to(device))
    
    return isp_train_p.detach().numpy().reshape(-1, 1)


def input_geatpy_cstar(x1, x2, x3, x4,smiles):
    X_input_Scaler = data_input(x1, x2, x3, x4,smiles)
    X_input_Scaler = torch.Tensor(X_input_Scaler.astype(np.float32))  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_cstar = Model_cstar_all([160,64,160,160,96,256,64,256,192,96,192,192,224])
    model_cstar.load_state_dict(torch.load('cstar_best_network.pth', 
                                         map_location=device))  
    
    model_cstar.eval()
    
    model_cstar = model_cstar.to(device)
    
    cstar_train_p = model_cstar(X_input_Scaler.to(device))
    
    return cstar_train_p.detach().numpy()

def input_geatpy_c_t(x1, x2, x3, x4,smiles):
    X_input_Scaler = data_input(x1, x2, x3, x4,smiles)
    X_input_Scaler = torch.Tensor(X_input_Scaler.astype(np.float32))  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_c_t = Model_c_t_all([160,128,192,256,128,224,192,224,192,128,128,160,32,192])
    model_c_t.load_state_dict(torch.load('c_t_best_network.pth', 
                                         map_location=device))  
    
    model_c_t.eval()
    
    model_c_t = model_c_t.to(device)
    
    c_t_train_p = model_c_t(X_input_Scaler.to(device))
    
    return c_t_train_p.detach().numpy()


