from tkinter import *
from tkinter.ttk import Notebook, PanedWindow, LabelFrame, Checkbutton
import tkinter.messagebox as messagebox
import math
from tkinter import *
from tkinter.ttk import *
from predict import input_geatpy_isp, input_geatpy_c_t, input_geatpy_cstar
from descriptors_generator import generator
import geatpy as ea  # import geatpy
from MyProblem_ import MyProblem
import numpy as np
import pandas as pd

def structure(x1, x2, x3, x4, smile):  
    c_t = input_geatpy_c_t(x1, x2, x3, x4, [smile])
    isp = input_geatpy_isp(x1, x2, x3, x4, [smile])
    cstar = input_geatpy_cstar(x1, x2, x3, x4, [smile])  
    var1.set(c_t[0][0])
    var2.set(isp[0][0])
    var3.set(cstar[0][0])

def GA(NIND_, MAXGEN_, XOVR_, smile, n1_max, n1_min, n2_max, n2_min, n3_max, n3_min, n4_max, n4_min):   
    problem = MyProblem(smile, n1_max, n1_min, n2_max, n2_min, n3_max, n3_min, n4_max, n4_min)
    Encoding = 'RI'  # Encoding method
    NIND = NIND_  # Population size
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # Create field descriptor  
    population = ea.Population(Encoding, Field, NIND)  # Instantiate population object (not yet initialized)

    """=========================== Algorithm Parameters ==========================="""
    myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # Instantiate algorithm template object
    myAlgorithm.MAXGEN = MAXGEN_  # Maximum generations
    myAlgorithm.mutOper.F = 0.6  # Differential evolution parameter F
    myAlgorithm.recOper.XOVR = XOVR_  # Crossover probability
    myAlgorithm.logTras = 1  # Log interval (0 means no logging)
    myAlgorithm.verbose = True  # Print log information
    myAlgorithm.drawing = 1  # Drawing mode (0: no drawing; 1: result plot; 2: animation in objective space; 3: animation in decision space)

    """========================== Run Algorithm ==============================="""
    [BestIndi, population] = myAlgorithm.run()  # Execute algorithm, get best individual and final generation population
    BestIndi.save()  # Save best individual information to file

    """================================= Output Results ======================="""
    print('Evaluation count: %s' % myAlgorithm.evalsNum)
    print('Elapsed time: %s seconds' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        output_text.delete(1.0, END)  # Clear previous text
        output_text.insert(END, f'Best objective function value: {BestIndi.ObjV[0][0]}\n')

        # Save to file
        with open('best_objective_value.txt', 'w') as f:
            f.write(f'Best objective function value: {BestIndi.ObjV[0][0]}\n')

        output_text.insert(END, 'Best control variable values:\n')
        result = []
        for i in range(BestIndi.Phen.shape[1]):
            result.append(BestIndi.Phen[0, i])

        # Save control variables to file and display
        with open('best_objective_value.txt', 'a') as f:
            f.write(f'AL: {result[0]}\n') 
            f.write(f'EMS: {result[1]}\n') 
            f.write(f'HTPB: {result[2]}\n') 
            f.write(f'AP: {result[3]}\n')

        # Display control variable details in the UI
        output_text.insert(END, f'AL: {result[0]}\n') 
        output_text.insert(END, f'EMS: {result[1]}\n') 
        output_text.insert(END, f'HTPB: {result[2]}\n') 
        output_text.insert(END, f'AP: {result[3]}\n')

    else:
        output_text.delete(1.0, END)
        output_text.insert(END, 'Finished\n')

# Function to run the program
def runprogram():
    try:
        x1 = float(entry1.get())
        x2 = float(entry2.get())
        x3 = float(entry3.get())
        x4 = float(entry4.get())
        flag = genetic_algorithm_var.get()
        smile = entry6.get()
    
        NIND_ = int(entry7.get())
        MAXGEN_ = int(entry8.get())
        XOVR_ = float(entry9.get())
        
        n1_max, n1_min = float(entry1_max.get()), float(entry1_min.get())
        n2_max, n2_min = float(entry2_max.get()), float(entry2_min.get())
        n3_max, n3_min = float(entry3_max.get()), float(entry3_min.get())
        n4_max, n4_min = float(entry4_max.get()), float(entry4_min.get())

        if flag == 1:  # If genetic algorithm is selected
            GA(NIND_, MAXGEN_, XOVR_, smile, n1_max, n1_min, n2_max, n2_min, n3_max, n3_min, n4_max, n4_min)
        else:
            structure(x1, x2, x3, x4, smile)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers!")

# Create main window
root = Tk()
root.title('AI Propellant Design')
root.geometry('1200x600')

notebook = Notebook(root)

frame1 = Frame()
notebook.add(frame1, text='AI Propellant Design')
notebook.pack(padx=5, pady=5, fill=BOTH, expand=True)

# Left panel for input
pw = PanedWindow(frame1, orient=HORIZONTAL)
pw.pack(padx=5, pady=5, fill=BOTH, expand=True)

# Main panels: Input, Genetic Algorithm, and Formula Information
pwin = PanedWindow(pw, orient=VERTICAL)
pw.add(pwin, weight=1)

# Create labeled frames
lf3 = LabelFrame(pwin, text='Input EMS', width=120)
lf4 = LabelFrame(pwin, text='Component Ratios', width=120)
lf5 = LabelFrame(pwin, text='Component Constraints', width=120)
lf6 = LabelFrame(pwin, text='Genetic Algorithm', width=120)
pwin.add(lf3, weight=1)
pwin.add(lf4, weight=1)
pwin.add(lf5, weight=1)
pwin.add(lf6, weight=1)

lf2 = LabelFrame(pw, text='Calculation Results', width=200)
pw.add(lf2, weight=1)

lf1 = LabelFrame(pw, text='Optimization Results', width=200)
pw.add(lf1, weight=1)

# Input section
structure_label1 = Label(lf3, text='SMILES Code')
entry6 = Entry(lf3, width=70)
entry6.insert(0, "[O-][N+](N1CN([N+]([O-])=O)CN([N+]([O-])=O)C1)=O")
structure_label1.grid(row=0, column=0, padx=5, pady=5)
entry6.grid(row=0, column=1, padx=5, pady=5)

structure_label6 = Label(lf4, text='AL Ratio')
entry1 = Entry(lf4, width=5)
entry1.insert(0, "18")
unit_label6 = Label(lf4, text='%')
structure_label6.grid(row=0, column=0, padx=5, pady=5)
entry1.grid(row=0, column=1, padx=5, pady=5)
unit_label6.grid(row=0, column=2, padx=5, pady=5)

structure_label7 = Label(lf4, text='EMS Ratio')
entry2 = Entry(lf4, width=5)
entry2.insert(0, "10")
unit_label7 = Label(lf4, text='%')
structure_label7.grid(row=0, column=3, padx=5, pady=5)
entry2.grid(row=0, column=4, padx=5, pady=5)
unit_label7.grid(row=0, column=5, padx=5, pady=5)

structure_label8 = Label(lf4, text='HTPB Ratio')
entry3 = Entry(lf4, width=5)
entry3.insert(0, "12")
unit_label8 = Label(lf4, text='%')
structure_label8.grid(row=1, column=0, padx=5, pady=5)
entry3.grid(row=1, column=1, padx=5, pady=5)
unit_label8.grid(row=1, column=2, padx=5, pady=5)

structure_label9 = Label(lf4, text='AP Ratio')
entry4 = Entry(lf4, width=5)
entry4.insert(0, "60")
unit_label9 = Label(lf4, text='%')
structure_label9.grid(row=1, column=3, padx=5, pady=5)
entry4.grid(row=1, column=4, padx=5, pady=5)
unit_label9.grid(row=1, column=5, padx=5, pady=5)

# More input fields
# structure_label11 = Label(lf4, text='Perform Formula Analysis')
# entry11 = Entry(lf4, width=5)
# entry11.insert(0, "0")
# structure_label11.grid(row=4, column=0, padx=5, pady=5)
# entry11.grid(row=4, column=1, padx=5, pady=5)

# Input fields for component ratio constraints
structure_label6_min = Label(lf5, text='AL Min Ratio')
entry1_min = Entry(lf5, width=5)
entry1_min.insert(0, "12")
structure_label6_min.grid(row=2, column=0, padx=5, pady=5)
entry1_min.grid(row=2, column=1, padx=5, pady=5)

structure_label6_max = Label(lf5, text='AL Max Ratio')
entry1_max = Entry(lf5, width=5)
entry1_max.insert(0, "18")
structure_label6_max.grid(row=2, column=2, padx=5, pady=5)
entry1_max.grid(row=2, column=3, padx=5, pady=5)

structure_label7_min = Label(lf5, text='EMS Min Ratio')
entry2_min = Entry(lf5, width=5)
entry2_min.insert(0, "20")
structure_label7_min.grid(row=3, column=0, padx=5, pady=5)
entry2_min.grid(row=3, column=1, padx=5, pady=5)

structure_label7_max = Label(lf5, text='EMS Max Ratio')
entry2_max = Entry(lf5, width=5)
entry2_max.insert(0, "50")
structure_label7_max.grid(row=3, column=2, padx=5, pady=5)
entry2_max.grid(row=3, column=3, padx=5, pady=5)

structure_label8_min = Label(lf5, text='HTPB Min Ratio')
entry3_min = Entry(lf5, width=5)
entry3_min.insert(0, "12")
structure_label8_min.grid(row=4, column=0, padx=5, pady=5)
entry3_min.grid(row=4, column=1, padx=5, pady=5)

structure_label8_max = Label(lf5, text='HTPB Max Ratio')
entry3_max = Entry(lf5, width=5)
entry3_max.insert(0, "14")
structure_label8_max.grid(row=4, column=2, padx=5, pady=5)
entry3_max.grid(row=4, column=3, padx=5, pady=5)

structure_label9_min = Label(lf5, text='AP Min Ratio')
entry4_min = Entry(lf5, width=5)
entry4_min.insert(0, "20")
structure_label9_min.grid(row=5, column=0, padx=5, pady=5)
entry4_min.grid(row=5, column=1, padx=5, pady=5)

structure_label9_max = Label(lf5, text='AP Max Ratio')
entry4_max = Entry(lf5, width=5)
entry4_max.insert(0, "50")
structure_label9_max.grid(row=5, column=2, padx=5, pady=5)
entry4_max.grid(row=5, column=3, padx=5, pady=5)

# Genetic algorithm input fields
input_volt1 = Label(lf6, text='Population Size')
entry7 = Entry(lf6, width=5)
entry7.insert(0, "1000")
unit_label5 = Label(lf6, text='NIND')
input_volt1.grid(row=0, column=0, padx=5, pady=5)
entry7.grid(row=0, column=1, padx=5, pady=5)
unit_label5.grid(row=0, column=2, padx=5, pady=5)

input_volt2 = Label(lf6, text='Max Generations')
entry8 = Entry(lf6, width=5)
entry8.insert(0, "500")
unit_label11 = Label(lf6, text='MAXGEN')
input_volt2.grid(row=1, column=0, padx=5, pady=5)
entry8.grid(row=1, column=1, padx=5, pady=5)
unit_label11.grid(row=1, column=2, padx=5, pady=5)

input_step = Label(lf6, text='Crossover Probability')
entry9 = Entry(lf6, width=5)
entry9.insert(0, "0.6")
unit_label12 = Label(lf6, text='XOVR')
input_step.grid(row=0, column=3, padx=5, pady=5)
entry9.grid(row=0, column=4, padx=5, pady=5)
unit_label12.grid(row=0, column=5, padx=5, pady=5)

# Use Checkbutton to enable/disable genetic algorithm
genetic_algorithm_var = IntVar()
genetic_algorithm_checkbox = Checkbutton(lf6, text="Enable Genetic Algorithm Optimization", variable=genetic_algorithm_var)
genetic_algorithm_checkbox.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

btn1 = Button(lf6, text='Run', command=runprogram)
btn1.grid(row=3, column=1, padx=5, pady=5)

# Output section
output_label1 = Label(lf2, text='c_t')
output_label1.grid(row=0, column=0, padx=5, pady=5)

output_label2 = Label(lf2, text='isp')
output_label2.grid(row=1, column=0, padx=5, pady=5)

output_label3 = Label(lf2, text='cstar')
output_label3.grid(row=2, column=0, padx=5, pady=5)

var1 = StringVar()
var1.set('             ')
msg1 = Label(lf2, textvariable=var1, relief="ridge")
msg1.grid(row=0, column=1, padx=5, pady=5)

var2 = StringVar()
var2.set('             ')
msg2 = Label(lf2, textvariable=var2, relief="ridge")
msg2.grid(row=1, column=1, padx=5, pady=5)

var3 = StringVar()
var3.set('             ')
msg3 = Label(lf2, textvariable=var3, relief="ridge")
msg3.grid(row=2, column=1, padx=5, pady=5)

output_text = Text(lf1, height=15, width=30)
output_text.grid(row=4, column=2, padx=5, pady=5)

# Run main window
root.mainloop()
