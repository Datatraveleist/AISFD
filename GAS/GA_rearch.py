from predict import input_geatpy_isp,input_geatpy_c_t,input_geatpy_cstar
import geatpy as ea  # import geatpy
from MyProblem_ import MyProblem
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('EOFCONH.csv')
formulation = []
X = data['smiles']
for idx, smiles in enumerate(X, start=1):
    print(f"Processing {idx}/{len(X)}: {smiles}")
    NIND_, MAXGEN_, XOVR_ = 800, 400, 0.6  # Population size, max generations, crossover probability  1000,400,0.6
    problem = MyProblem(smiles)
    Encoding = 'RI'  # Encoding method
    NIND = NIND_  # Population size
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # Create field descriptor
    population = ea.Population(Encoding, Field, NIND)  # Instantiate population object

    # Algorithm parameter settings
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # Instantiate algorithm template
    myAlgorithm.MAXGEN = MAXGEN_  # Max generations
    myAlgorithm.mutOper.F = 0.6  # Differential evolution parameter F
    myAlgorithm.recOper.XOVR = XOVR_  # Crossover probability
    myAlgorithm.logTras = 1  # Log every generation
    myAlgorithm.verbose = True  # Print log info
    myAlgorithm.drawing = 0  # No drawing

    # Run the algorithm
    [BestIndi, population] = myAlgorithm.run()  # Execute algorithm, get best individual and final population
    BestIndi.save()  # Save best individual info

    if BestIndi.sizes != 0:
        print(f"Best objective value: {BestIndi.ObjV[0][0]}")
        print("Best control variable values:")
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])

        c_t = input_geatpy_c_t(BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3], [smiles])
        isp = input_geatpy_isp(BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3], [smiles])
        cstar = input_geatpy_cstar(BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3], [smiles])
    else:
        print("No valid solution found.")

    formulation.append([smiles, c_t[0], isp[0][0], cstar[0], BestIndi.Phen[0, 0], BestIndi.Phen[0, 1], BestIndi.Phen[0, 2], BestIndi.Phen[0, 3]])
formulation = pd.DataFrame(formulation,columns = ['smiles','c_t','isp','cstar','Al', 'EMs', 'HTPB','NH4CLO4'])
formulation.to_csv('formulations_number3_NIND_800.csv', index=False)  

