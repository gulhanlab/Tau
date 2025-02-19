from sage.all import load, var, solve
import sys
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def get_solutions(major, minor):
    #get only solution files compatible with copy number state 
    pattern = f"{major}_{minor}.*_solutions.sobj"
    sols = [solutions_dir + '/' + re.match(pattern, sol).group() for sol in all_solutions if re.match(pattern, sol) is not None]
    return sols

def extract_coefficients_from_constraints(constraints, variables):
    coefficients = []
    for constraint in constraints:
        lhs = constraint.left_hand_side()
        coeffs = [lhs.coefficient(var) for var in variables]
        coefficients.append(coeffs) 
    return coefficients

def neg_log_likelihood(lambdas, counts):
    return -np.sum(counts * np.log(lambdas) - lambdas)

#changing sage inequalities
def change_inequality(ineqs):
    new_ineqs = []
    for ineq in ineqs:
        if ineq.operator() == operator.gt:
            new_ineqs.append(ineq.lhs() >= ineq.rhs())  # Change > to >=
        elif ineq.operator() == operator.lt:
            new_ineqs.append(ineq.lhs() <= ineq.rhs())  # Change < to <=
        else:
            new_ineqs.append(ineq)
    return new_ineqs

#for dealing with floating point errors
def round_small_values(expr, tol=1e-7):
    if expr.is_relational():  # If the expression is an inequality or equation
        return expr.operator()(round_small_values(expr.lhs(), tol), round_small_values(expr.rhs(), tol))
    elif expr.operator() is None:  # If the expression is a number or variable
        return 0 if abs(expr) < tol else expr
    else:  # Recursively apply to function arguments
        return expr.operator()(*[round_small_values(arg, tol) for arg in expr.operands()])

def parse_chromosome(chrom):
    # If chromosome is numeric, return as int; otherwise return a large number for sorting
    if chrom.isdigit():
        return int(chrom)
    # Assign an arbitrary large number to non-numeric chromosomes for proper sorting
    # X = 23, Y = 24, MT = 25 (for example)
    return {"X": 23, "Y": 24, "MT": 25}.get(chrom, 26)  # Default to 26 for unhandled cases

sorted_segments = dict(
    sorted(
        segment_solutions.items(),
        key=lambda item: (parse_chromosome(item[0].split(':')[0]), int(item[0].split(':')[1].split('-')[0]))
    )
)

def calculate_timing_solutions(multiplicities_file=None, output_name=None, multiplicities_df=None):
    if multiplicities_df is None:
        multiplicities_df = pd.read_csv(multiplicities_file, sep='\t')

    copy_number_dict = dict(zip(multiplicities_df['segment_id'],
                               multiplicities_df['major_cn'].astype(str)+'_' + multiplicities_df['minor_cn'].astype(str)))
    signatures = ["SBS1"]
    signature_tag = '_'.join(signatures)
    SBS_df = multiplicities_df.query('sig_max in @signatures')

    solutions_dir = '/n/data1/hms/dbmi/park/jbrew/matrices/new_solutions' #CHANGE TO DOWNSIZED WHEN COMPLETE
    all_solutions = os.listdir(solutions_dir)

    segment_solutions = defaultdict(lambda: defaultdict(dict))

    for idx, row in SBS_df.iterrows():
        #calculate segment by segment timing results
        segment = row['segment_id']
        #print("SEGMENT: ", segment)
        major = row['major_cn']
        minor = row['minor_cn']
        N_vars = [f'N{i}' for i in range(1, major+1)]
        t_num = major + minor - 1 if minor > 0 else major
        if t_num == 0:
            continue
        t_vars = [f't{i}' for i in range(1, t_num+1)]
        var(N_vars)
        var(t_vars)
        N_values = {var(N_var): N_val for N_var, N_val in row.filter(items = N_vars).items()}
        if sum(N_values.values()) < 5:
            continue

        max_in_row = max(int(num[1:]) for num in row.filter(regex='N.*').keys().to_list())

        #this happens if there are zero values for some high multiplicities so we have to fill in these zeros
        if major > max_in_row:
            for i in range(max_in_row+1, major+1):
                N_values[var(f'N{i}')] = 0

        sols = get_solutions(major, minor) 
        unfit = defaultdict(list)
        unfit_constraints = defaultdict(list)

        for sol_file in sols:
            #if solution_found:
            #    break
            #print('SOLUTIONS FILE:', sol_file)
            #load solutions
            eqs, ineqs, solutions = load(sol_file)
            sol_num = len(solutions)
            segment_solutions[segment][sol_file] = list()
        
            for i, solution in enumerate(solutions):
                if i != (len(solutions) - 1):
                    continue
                substitution = change_inequality([s.subs(N_values) for s in solution])
                no_variable = [sub for sub in substitution if len(sub.variables())==0]
                variable = [sub for sub in substitution if len(sub.variables())> 0]
                if all(no_variable):
                    #solution_found = True
                    #split into equalities and inequalities
                    equalities = [var for var in variable if var.operator() == operator.eq]
                    inequalities = [var for var in variable if var.operator() in [operator.ge, operator.le]]
                    #if we assume the inequalities come from dependent variables then this would be really really good
        
                    #identify dependent and independent variables
                    dep = [eq.lhs() for eq in equalities] # will this work? think so... will revisit if not
                    indep = list(set(var(t_vars)) - set(dep)) 
        
                    equalities = {var.lhs(): var.rhs() for var in equalities}
        
                    #split equalities into constants and non-constants
                    constants = {var: value for var, value in equalities.items() if len(value.variables()) == 0}
                    nonconstants = {var: value for var, value in equalities.items() if len(value.variables()) > 0}
        
        
                    #classifying minimum and maximum values for independent variables
                    indep_min_max = {}
        
                    for ind in indep:
                        relevant_ineqs = [ineq for ineq in inequalities if ind in ineq.variables()]
                        operators = [ineq.operator() for ineq in inequalities if ind in ineq.variables()]
                        indep_min = float('inf')
                        indep_max = -float('inf')
                        for ineq, _operator in zip(relevant_ineqs, operators):
                            if not len(ineq.lhs().variables()):
                                if _operator == operator.le:
                                    indep_min = np.minimum(indep_min, ineq.lhs())
                                else:
                                    indep_max = np.maximum(indep_max, ineq.lhs())
                            else:
                                if _operator == operator.ge:
                                    indep_min = np.minimum(indep_min, ineq.rhs())
                                else:
                                    indep_max = np.maximum(indep_max, ineq.rhs())
        
                        indep_min_max[ind] = (indep_min, indep_max)
                        
        
                    #if t_num > 1 and i+1 == sol_num:
        
                    curr_sol = {'constants': constants, 
                            'non-constants': nonconstants,
                            'min_max': indep_min_max}
        
                    #if len(inequalities) > 0: 
                        #print(f"{major}_{minor}, with {N_vars} and {t_vars}")
                        #print("N values:", N_values)
                        #print("Equations:", eqs)
                        #print(solution)
                        #print('equalities:', equalities)
                        #print('inequalities:', inequalities)
                        #print('min max:', indep_min_max)
                        #print('dependent variables:', dep)
                        #print('independent variables:', indep)
                        #print('constants', constants)
                        #print('non-constants', nonconstants)
                    #fit[sol_file].append(solution)
                    segment_solutions[segment][sol_file].append(curr_sol)  
                else:
                    constraints = [x for x in solution if all([var(t_var) not in x.variables() for t_var in t_vars])]
                    unfit[sol_file].append(solution)
                    unfit_constraints[sol_file].append(constraints)

        ## DEALING WITH MLE estimates
        if sum([len(x) for x in segment_solutions[segment].values()]) == 0:
            print('NO SOLUTION! Calculating MLE estimate')
            unfit_likelihoods = {}
            unfit_lambdas = {}
            #segment_solutions[segment]['no_solution'] = True
        
            #ML estimates
            for name, sol in unfit_constraints.items():
                sage_constraints = sol[-1]
                observed_counts = N_values
                sage_constraints_inequalities = [x for x in sage_constraints if x.operator() != operator.eq]  
                sage_constraints_equalities = [x for x in sage_constraints if x.operator() == operator.eq]  
                
                inequality_coefficients = extract_coefficients_from_constraints(sage_constraints_inequalities, var(N_vars))
                equality_coefficients = extract_coefficients_from_constraints(sage_constraints_equalities, var(N_vars))
                
                constraints = [
                    {'type': 'ineq', 'fun': lambda x, c=coeffs: np.dot(c, x)}  # c⋅x >= 0
                    for coeffs in inequality_coefficients
                ] + [{'type': 'eq', 'fun': lambda x, c=coeffs: np.dot(c,x)}
                    for coeffs in equality_coefficients]
            
                observed_counts = list(N_values.values())
                initial_guess = np.mean(observed_counts) # A good starting point for each lambda
                initial_guess = np.full(len(observed_counts), initial_guess)
                
                result = minimize(neg_log_likelihood, initial_guess, args=(observed_counts,), constraints=constraints)
                
                mle_lambdas = result.x  # MLE for each lambda
                #print(sage_constraints)
                max_likelihood = np.prod(poisson.pmf(observed_counts, mle_lambdas))
                
                unfit_likelihoods[name] = max_likelihood
                unfit_lambdas[name] = mle_lambdas
        
            sol_file = max(unfit_likelihoods, key=unfit_likelihoods.get)
            #print("MLE SOLUTION:", sol_file)
            #print(unfit_lambdas[sol_file])
            #print(unfit_likelihoods[sol_file])
            #print(unfit_likelihoods)
            eqs, ineqs, solutions = load(sol_file)
            sol_num = len(solutions)
            solution = solutions[-1]
            
            N_values = {N_val: val for N_val, val in zip(N_values.keys(), np.round(unfit_lambdas[sol_file], decimals=8))}
            substitution = change_inequality([s.subs(N_values) for s in solution])
            no_variable = [round_small_values(sub) for sub in substitution if len(sub.variables())==0]
            variable = [sub for sub in substitution if len(sub.variables())> 0]
            #print(segment, sol_file)
            #print(no_variable)
            if not all(no_variable):
                print("FAILED")
                print(no_variable)
                break
            if all(no_variable):
                print("PASSED!")
                #solution_found = True
                #split into equalities and inequalities
                equalities = [var for var in variable if var.operator() == operator.eq]
                inequalities = [var for var in variable if var.operator() in [operator.le, operator.ge]]
                #if we can assume the inequalities come from dependent variables then this would be really really good
                
                #identify dependent and independent variables
                dep = [eq.lhs() for eq in equalities] # will this work? think so... will revisit if not
                indep = list(set(var(t_vars)) - set(dep)) 
                
                equalities = {var.lhs(): var.rhs() for var in equalities}
                
                #split equalities into constants and non-constants
                constants = {var: round_small_values(value) for var, value in equalities.items() if len(value.variables()) == 0}
                nonconstants = {var: round_small_values(value) for var, value in equalities.items() if len(value.variables()) > 0}
                
                #classifying minimum and maximum values for independent variables
                indep_min_max = {}
                
                for ind in indep:
                    relevant_ineqs = [round_small_values(ineq) for ineq in inequalities if ind in ineq.variables()]
                    operators = [ineq.operator() for ineq in relevant_ineqs]
                    indep_min = float('inf')
                    indep_max = -float('inf')
                    print(relevant_ineqs, operators)
                    for ineq, _operator in zip(relevant_ineqs, operators):
                        if not len(ineq.lhs().variables()):
                            if _operator == operator.le:
                                indep_min = np.minimum(indep_min, ineq.lhs())
                            else:
                                indep_max = np.maximum(indep_max, ineq.lhs())
                        else:
                            if _operator == operator.ge:
                                indep_min = np.minimum(indep_min, ineq.rhs())
                            else:
                                indep_max = np.maximum(indep_max, ineq.rhs())
                    indep_min_max[ind] = (indep_min, indep_max)
                curr_sol = {'constants': constants, 
                        'non-constants': nonconstants,
                        'min_max': indep_min_max}
                print(curr_sol)
                #if len(inequalities) > 0: 
                    #print(f"{major}_{minor}, with {N_vars} and {t_vars}")
                    #print("N values:", N_values)
                    #print("Equations:", eqs)
                    #print(solution)
                    #print('equalities:', equalities)
                    #print('inequalities:', inequalities)
                    #print('min max:', indep_min_max)
                    #print('dependent variables:', dep)
                    #print('independent variables:', indep)
                    #print('constants', constants)
                    #print('non-constants', nonconstants)
                #fit[sol_file].append(solution)
                segment_solutions[segment][sol_file].append(curr_sol)    

    sorted_segments = dict(sorted(
        segment_solutions.items(),
        key=lambda item: (parse_chromosome(item[0].split(':')[0]), int(item[0].split(':')[1].split('-')[0]))))
    return sorted_segments

def process_solutions(segment_solutions):
    # Collect all unique t variables across the dataset
    all_t_vars = set()
    for region_data in data.values():
        for entries in region_data.values():
            for entry in entries:
                all_t_vars.update(entry.get('constants', {}).keys())
                all_t_vars.update(entry.get('non-constants', {}).keys())
                all_t_vars.update(entry.get('min_max', {}).keys())

    # Sort t variables numerically (t1, t2, t3, ...)
    sorted_t_vars = sorted(all_t_vars, key=lambda x: int(str(x)[1:])) if all_t_vars else []
    max_t = len(sorted_t_vars)
    rows = []
    
    for region_key, region_data in data.items():
        chrom, positions = region_key.split(':')
        start, end = positions.split('-')
        
        for file_path, entries in region_data.items():
            if not entries:
                row = {
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'file_path': file_path,
                    'fully_solved': False,
                    'no_solution': True
                }
                for t_var in sorted_t_vars:
                    row[t_var] = np.nan
                rows.append(row)
            else:
                for entry in entries:
                    row = {
                        'chromosome': chrom,
                        'start': start,
                        'end': end,
                        'file_path': file_path
                    }
                    
                    present_vars = set()
                    var_types = {}
                    
                    # Process each variable type
                    for t_var in sorted_t_vars:
                        if t_var in entry.get('constants', {}):
                            val = entry['constants'][t_var]
                            row[t_var] = float(val) if isinstance(val, (int, float)) else val
                            present_vars.add(t_var)
                            var_types[t_var] = 'constant'
                        elif t_var in entry.get('min_max', {}):
                            mm = entry['min_max'][t_var]
                            row[t_var] = mm
                            present_vars.add(t_var)
                            var_types[t_var] = 'range'
                        elif t_var in entry.get('non-constants', {}):
                            row[t_var] = entry['non-constants'][t_var]
                            present_vars.add(t_var)
                            var_types[t_var] = 'expression'
                        else:
                            row[t_var] = np.nan

                    # Calculate cumulative sums where possible
                    cumulative = 0
                    valid_cumulative = True
                    for k in range(1, max_t + 1):
                        t_key = var(f't{k}')
                        if t_key not in present_vars:
                            valid_cumulative = False
                        elif var_types.get(t_key) != 'constant':
                            valid_cumulative = False
                        
                        if valid_cumulative:
                            cumulative += row[t_key]
                            row[f'sum_t1_to_t{k}'] = cumulative
                        else:
                            row[f'sum_t1_to_t{k}'] = np.nan

                    # Determine if fully solved
                    row['fully_solved'] = all(
                        var_types.get(t_var) == 'constant'
                        for t_var in present_vars
                    ) if present_vars else False
                    row['no_solution'] = False
                    rows.append(row)

    # Create DataFrame with consistent column order
    standard_cols = ['chromosome', 'start', 'end', 'file_path',]
    sum_cols = [f'sum_t1_to_t{k}' for k in range(1, max_t+1)] if max_t > 0 else []
    final_df = pd.DataFrame(rows)[standard_cols + sorted_t_vars + sum_cols + ['fully_solved', 'no_solution']]
    final_df = final_df.rename(columns={variable: str(variable) for variable in final_df.filter(regex=r'^t\d+').columns})
    final_df['tree_structure'] = [os.path.basename(x).split('_solution')[0] for x in final_df['file_path']]
    del final_df['file_path']
    return final_df 

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest
import matplotlib.pyplot as plt
import re
from scipy.spatial import KDTree

def calculate_breakpoints(sorted_segments):
    all_segments = defaultdict(dict)

    for segment, data in sorted_segments.items():
        for sol_name, solutions in data.items():
            for solution in solutions:
                all_segments[segment][sol_name] = {}
                min_max = solution.get("min_max", {})
                if len(min_max) <= 1:
                    if len(min_max) == 0:
                        constants = solution.get("constants", {})
                        y_vals = constants
                        all_segments[segment][sol_name]['vals'] = (None, y_vals)
                        all_segments[segment][sol_name]['constant'] = True
                    
                    elif len(min_max) == 1:  # Only process single independent variable segments
                        var, bounds = list(min_max.items())[0]
                        lower, upper = bounds
                        x_vals = np.linspace(float(lower), float(upper), 3) 
                        non_constants = solution.get("non-constants", {})
                        constants = solution.get("constants", {})
                        y_vals = {}
                        for dep_variable, dep_value in non_constants.items():
                            y_vals[dep_variable] = np.array([float(dep_value.subs({var: val})) for val in x_vals])
                        for dep_variable, dep_value in constants.items():
                            y_vals[dep_variable] = np.array([float(dep_value) for val in x_vals])
                        y_vals[var] = x_vals
                        
                        all_segments[segment][sol_name]['vals'] = (x_vals, y_vals)
                        all_segments[segment][sol_name]['constant'] = bool(upper == lower)

    all_t_vars = set()
    for region_data in all_segments.values():
        for entries in region_data.values():
            if 'vals' not in entries:
                continue
            for entry in entries['vals'][1]:
                all_t_vars.add(entry)

    # Sort t variables numerically (t1, t2, t3, ...)
    sorted_t_vars = sorted(all_t_vars, key=lambda x: int(str(x)[1:])) if all_t_vars else []
    max_t = len(sorted_t_vars)

    rows = []
    for segment_id, data in all_segments.items():
        chrom, positions = segment_id.split(':')
        start, end = positions.split('-')
        for sol_name, solution in data.items():
            if 'vals' not in solution:
                row = {
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'file_path': sol_name,
                    'averaged': float(np.nan),
                    'solved': False
                        }
            else:
                row = {
                        'chromosome': chrom,
                        'start': start,
                        'end': end,
                        'file_path': sol_name
                            }
                for t_var in sorted_t_vars:
                    y_vals = solution['vals'][1]
                    row[t_var] = np.median(y_vals[t_var]) if t_var in y_vals else np.nan
                row['averaged'] = not solution['constant']
                row['solved'] = True
            rows.append(row)

    average_df = pd.DataFrame(rows)

    t_cols = average_df.filter(regex=r't\d').astype(float)
    normalized = t_cols.apply(lambda x: x / np.sum(x), axis=1)

    normalized_cumsum = normalized.apply(np.cumsum, axis=1)
    normalized_cumsum = normalized_cumsum.rename(columns=lambda col: f"{col}_norm_cumsum")

    normalized = normalized.rename(columns=lambda col: f"{col}_norm")
    average_df = pd.concat([average_df, normalized, normalized_cumsum], axis=1)
    average_df['major'] = average_df['file_path'].apply(lambda x: re.split(r'_|\.', os.path.basename(x))[0])
    average_df['minor'] = average_df['file_path'].apply(lambda x: re.split(r'_|\.', os.path.basename(x))[2])
    average_df.columns = [str(x) for x in list(average_df.columns)]
    
    # Extract timing and sum columns
    t_columns = [col for col in average_df.columns if re.match(r'^t\d+$', col)]
    sum_columns = [col for col in average_df.columns if re.match(r'.*norm_cumsum.*', col)]
    df = average_df
    breakpoint_columns = sum_columns
    df.loc[:, 'breakpoints'] = df.loc[:, breakpoint_columns].apply(lambda row: row.dropna().tolist(), axis=1)

    # Extract breakpoints (cumulative sums)
    df['breakpoints'] = df.loc[:, breakpoint_columns].apply(lambda row: row.dropna().tolist(), axis=1)

    # Pool all breakpoints across segments
    all_breakpoints = [x for x in df.loc[:, 'breakpoints'] if len(x) > 1]  # Exclude segments with < 2 breakpoints

    if len(all_breakpoints) > 0:
        all_breakpoints = np.concatenate(all_breakpoints)  # Flatten into a single array
        all_breakpoints = [x for x in all_breakpoints if x > 0 and x < 1]  # Exclude breakpoints at boundaries
    else:
        all_breakpoints = np.nan

    return all_breakpoints

def plot_timing_results(all_segments, median_breakpoint):
    ### PLOTTING SEGMENT TIMINGS
    average = False
    fig, ax = plt.subplots(figsize=(30, 15))

    # Define a list of colormaps to use for gradations
    colormaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, 
                 plt.cm.Oranges, plt.cm.Purples, plt.cm.Greys,
                 plt.cm.YlGnBu, plt.cm.BuPu, plt.cm.GnBu, 
                 plt.cm.PuRd, plt.cm.coolwarm, plt.cm.Spectral, 
                 plt.cm.PiYG, plt.cm.BrBG, plt.cm.viridis, 
                 plt.cm.plasma, plt.cm.cividis, plt.cm.magma, plt.cm.inferno]

    # Function to generate gradations for a given number of t values
    def get_gradated_colors(num_t_values, colormap):
        return [colormap(i / num_t_values) for i in range(1, num_t_values + 1)]

    # Dictionary to store legend handles and labels for each colormap
    legend_data = {cmap.name: {"handles": [], "labels": []} for cmap in colormaps}

    chrom = '1'
    offset = 0
    old_segment_end = 0
    segment_xpos = -0.02
    for segment_name, values in all_segments.items():
        line = False
        valid_sol_idx = np.where(['vals' in x for x in values.values()])[0]
        choice = np.random.choice(valid_sol_idx)
        choice = list(values.keys())[choice]
        segment = values[choice]  # pick most general solution (last solution!)
        segment_start = int(segment_name.split(':')[1].split('-')[0])
        segment_end = int(segment_name.split(':')[1].split('-')[1])
        new_chrom = segment_name.split(':')[0]
        
        if new_chrom != chrom:
            line = True
            chrom = new_chrom
        
        if line:
            ax.vlines(offset, ymin=0, ymax=1, color='black', linestyle='--',
                     linewidth=3)

        if segment['vals'][0] is not None and not average:
            x_vals, y_vals = segment['vals']
            y_vals = {str(key): val for key, val in list(y_vals.items())}
            y_vals = dict(sorted(y_vals.items(), key=lambda x: int(x[0][1:])))
            cumulative_bottom = np.zeros(len(x_vals))  # Start stacking from the bottom

            # Compute the total sum at each index across all variables
            total_sum_per_index = np.sum(
                np.array([values for values in y_vals.values()], dtype=float),
                axis=0,
            )
            x_vals = np.linspace(0, 1, 3)

            # Determine the number of t values and select the appropriate colormap
            num_t_values = len(y_vals)
            colormap = colormaps[(num_t_values - 1) % len(colormaps)]  # Cycle through colormaps
            colors = get_gradated_colors(num_t_values, colormap)

            for idx, (variable, values) in enumerate(y_vals.items()):
                numeric_values = np.array(values, dtype=float)  # Ensure numeric
                normalized_values = numeric_values / total_sum_per_index

                legend_label = f"t{idx + 1}/{num_t_values}"  # e.g., t1/2, t2/2, t1/3, etc.

                # Plot the segment
                fill = ax.fill_between(
                    x_vals * 91913600 + offset,
                    cumulative_bottom,
                    cumulative_bottom + normalized_values,
                    color=colors[idx],
                    alpha=0.8,
                )

                # Store the handle and label for the legend
                if legend_label not in legend_data[colormap.name]["labels"]:
                    legend_data[colormap.name]["handles"].append(fill)
                    legend_data[colormap.name]["labels"].append(legend_label)

                cumulative_bottom += normalized_values  # Stack for the next variable

            segment_center = offset + 91913600 / 2  # Center of the segment
            ax.text(
                segment_center, 
                segment_xpos,  # Slightly below the y-axis range
                segment_name, 
                rotation=45, 
                fontsize=14, 
                verticalalignment='top', 
                horizontalalignment='right',
                rotation_mode='anchor' 
            )
            ax.vlines(segment_center, 0, -0.015, color='black')
            ax.text(segment_center, 1.02, copy_number_dict[segment_name], 
                    fontsize=15, rotation=90,
                   verticalalignment='center',horizontalalignment='center')
            offset += 91913600

        elif segment['vals'][1] is not None or average:  # For constants
            y_vals = segment['vals'][1]
            y_vals = {str(key): np.median(val) for key, val in list(y_vals.items())}
            y_vals = dict(sorted(y_vals.items(), key=lambda x: int(x[0][1:])))
            x_vals = np.linspace(0, 1, 3)
            cumulative_bottom = np.zeros(len(x_vals))  # Start stacking from the bottom

            # Compute the total sum of constant values
            total_sum = sum([float(value) for value in y_vals.values()])

            # Determine the number of t values and select the appropriate colormap
            num_t_values = len(y_vals)
            colormap = colormaps[(num_t_values - 1) % len(colormaps)]  # Cycle through colormaps
            colors = get_gradated_colors(num_t_values, colormap)

            for idx, (variable, value) in enumerate(y_vals.items()):
                numeric_value = float(value)
                normalized_value = numeric_value / total_sum if total_sum > 0 else 0
                constant_values = np.full(len(x_vals), normalized_value, dtype=float)

                # Generate the legend label dynamically
                if num_t_values == 1:
                    legend_label = "t1/1"  # Special case for 1 t value
                else:
                    legend_label = f"t{idx + 1}/{num_t_values}"  # e.g., t1/2, t2/2, t1/3, etc.

                # Plot the segment
                fill = ax.fill_between(
                    offset + x_vals * 91913600,
                    cumulative_bottom,
                    cumulative_bottom + constant_values,
                    color=colors[idx],
                    alpha=0.8,
                )

                # Store the handle and label for the legend
                if legend_label not in legend_data[colormap.name]["labels"]:
                    legend_data[colormap.name]["handles"].append(fill)
                    legend_data[colormap.name]["labels"].append(legend_label)

                cumulative_bottom += constant_values  # Stack for the next constant variable

            segment_center = offset + 91913600 / 2  # Center of the segment
            ax.text(
                segment_center, 
                segment_xpos,
                segment_name, 
                rotation=45, 
                fontsize=14, 
                verticalalignment='top', 
                horizontalalignment='right',
                rotation_mode='anchor'
            )
            ax.vlines(segment_center, 0, -0.015, color='black')
            ax.text(segment_center, 1.02, copy_number_dict[segment_name], 
                    fontsize=15, rotation=90,
                   verticalalignment='center',horizontalalignment='center')
            offset += 91913600

    # Add legends for each colormap, placed to the right of the plot
    legend_spacing = 0.23 # Adjust this to control the spacing between legends
    for i, cmap in enumerate(colormaps):
        if legend_data[cmap.name]["handles"]:  # Only add a legend if there are handles
            legend = ax.legend(
                legend_data[cmap.name]["handles"],
                legend_data[cmap.name]["labels"],
                title=f"n = {i+1}",
                loc="center left",
                fontsize=15,title_fontsize = 15,
                bbox_to_anchor=(.955, 1.2 - legend_spacing),  # Place legends to the right
                frameon=False
            )
            legend_spacing = legend_spacing * 1.4
            ax.add_artist(legend)  # Add the legend to the plot

    # Adjust the figure layout to make space for the legends
    plt.subplots_adjust(right=0.7)  # Increase the right margin to fit the legends
    ax.hlines(median_breakpoint, xmin = 0, xmax = offset, color='red', 
              linestyle = '--', linewidth=3)
    ax.set_xlabel("")
    ax.set_ylabel("Mutation time", fontsize=20)
    average_tag = "" if not average else "_AVERAGED"
    ax.set_title(f"{output_name}{average_tag}", fontsize=30) 
    ax.grid(True)
    plt.ylim(-0.1999,1.05)
    ax.set_xticks([])
    plt.tight_layout()
    #plt.savefig(f"{directory}/{output_name}_segment_plot_{signature_tag}.png", dpi=300)
    plt.show()
    plt.close()
