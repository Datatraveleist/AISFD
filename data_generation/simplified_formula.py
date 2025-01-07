import numpy as np
import pandas as pd

# Molar masses of elements in g/mol
molar_masses = {
    'C': 12.01,
    'H': 1.008,
    'O': 16.00,
    'N': 14.01,
    'Al': 26.98,
    'Cl': 35.45
}

def get_components(row):
    """
    Extracts components, their compositions, and molecular weights from a row of the DataFrame.
    """
    components = {
        'Al': row['Al'] / 100,  
        'EMs': row['EMs'] / 100,  
        'HTPB': row['HTPB'] / 100,  
        'NH4CLO4': row['NH4CLO4'] / 100  
    }
    element_composition = {
        'NH4CLO4': {'H': 4, 'O': 4, 'N': 1, 'Cl': 1},
        'EMs': {'C': row['nC'], 'H': row['nH'], 'O': row['nO'], 'N': row['nN']},
        'Al': {'Al': 1},
        'HTPB': {'C': 10, 'H': 15.40, 'O': 0.07}
    }
    molecular_weight = {
        'Al': 26.98,
        'EMs': row['molecular weight'],
        'HTPB': 86,
        'NH4CLO4': 117.49
    }
    return components, element_composition, molecular_weight

def calculate_imaginary_formula(components, element_composition, molecular_weight, molar_masses, total_mass=1000):
    """
    Calculates the total moles of each element in the imaginary formula.
    """
    total_moles = {element: 0 for element in molar_masses.keys()}  # Initialize moles for each element

    for component, percentage in components.items():
        component_mass = total_mass * percentage  # Calculate mass of the component
        mol = component_mass / molecular_weight[component]  # Calculate moles of the component
        for element, count in element_composition.get(component, {}).items():
            total_moles[element] += count * mol  # Accumulate moles for each element

    return total_moles

# Load input data
df_train = pd.read_csv('data_train.csv')
result = []

# Iterate through each row of the DataFrame
for _, row in df_train.iterrows():
    components, element_composition, molecular_weight = get_components(row)
    total_moles = calculate_imaginary_formula(components, element_composition, molecular_weight, molar_masses)
    # Append the moles of each element to the results
    result.append([total_moles['C'], total_moles['H'], total_moles['O'], total_moles['N'], total_moles['Al'], total_moles['Cl']])

# Create a DataFrame from the results
df_result = pd.DataFrame(result, columns=['C_mol', 'H_mol', 'O_mol', 'N_mol', 'Al_mol', 'Cl_mol'])

# Save the result to a CSV file
df_result.to_csv('ECs123test.csv', index=False)
print("Calculation complete. Results saved to 'ECs123test.csv'.")
