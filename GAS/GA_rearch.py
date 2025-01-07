from predict import input_geatpy_isp, input_geatpy_c_t, input_geatpy_cstar
import geatpy as ea  # import geatpy
from MyProblem_ import MyProblem
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Read input data from CSV file
data = pd.read_csv('EOFCONH.csv')
formulation = []

# Extract smiles data
X = data['smiles']

# Iterate over each smiles string
for smiles in X:
    NIND_, MAXGEN_, XOVR_ = 1000, 400, 0.6  # Population size, maximum generations, and crossover probability
    problem = MyProblem(smiles)  # Define the optimization problem

    Encoding = 'RI'  # Encoding method
    NIND = NIND_  # Number of individuals
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # Create field
    population = ea.Population(Encoding, Field, NIND)  # Initialize population object

    """=========================== Algorithm parameter settings =========================="""
    # Instantiate algorithm object (population not initialized yet)
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # Instantiate a DE algorithm template
    
    myAlgorithm.MAXGEN = MAXGEN_  # Maximum number of generations
    myAlgorithm.mutOper.F = 0.6  # Differential evolution parameter F
    myAlgorithm.recOper.XOVR = XOVR_  # Set crossover probability
    myAlgorithm.logTras = 1  # Set how frequently logs are recorded; 0 means no logs
    myAlgorithm.verbose = True  # Set whether to output log information
    myAlgorithm.drawing = 0  # Set drawing method (0: no drawing, 1: result plots, 2: target space animation, 3: decision space animation)
    
    """========================== Run algorithm for population evolution ==============="""
    [BestIndi, population] = myAlgorithm.run()  # Run the algorithm and obtain the best individual and the final generation population
    BestIndi.save()  # Save the information of the best individual

    """============================== Output results ==============================="""
    if BestIndi.sizes != 0:  # If a valid solution is found
        print('The optimal objective function value is: %s' % BestIndi.ObjV[0][0])
        print('The optimal control variable values are:')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
        
        # Predict c_t, isp, and cstar using the respective models
        c_t = input_geatpy_c_t(BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3], [smiles])
        isp = input_geatpy_isp(BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3], [smiles])
        cstar = input_geatpy_cstar(BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3], [smiles])
        
    else:
        print('Optimization finished with no valid solution.')

    # Append the results to the formulation list
    formulation.append([smiles, c_t[0], isp[0][0], cstar[0], BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3]])

# Convert the results into a DataFrame and save as CSV
formulation = pd.DataFrame(formulation, columns=['smiles', 'c_t', 'isp', 'cstar', 'Al', 'EMs', 'HTPB', 'NH4CLO4'])
formulation.to_csv('formulations.csv', index=False)
