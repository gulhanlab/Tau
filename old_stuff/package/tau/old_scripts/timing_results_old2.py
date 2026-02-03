from sage.all import load, var, solve
import sys
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import ranksums 
import time
from scipy.optimize import minimize
import math
from scipy.stats import poisson
from scipy.stats import mannwhitneyu, binomtest
import matplotlib.pyplot as plt
import re
from scipy.spatial import KDTree
import operator
from io import StringIO

#helper functions
def get_solutions(major, minor, solutions_dir):
    #get only solution files compatible with copy number state 
    all_solutions = os.listdir(solutions_dir)
    pattern = f"{major}_{minor}.*.sobj"
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
def round_small_values(expr, tol=1e-3):
    if expr.is_relational():  # If the expression is an inequality or equation
        return expr.operator()(round_small_values(expr.lhs(), tol), round_small_values(expr.rhs(), tol))
    elif expr.operator() is None:  # If the expression is a number or variable
        return 0 if abs(expr) < tol else expr
    else:  # Recursively apply to function arguments
        return expr.operator()(*[round_small_values(arg, tol) for arg in expr.operands()])

def parse_chromosome(chrom):
    if 'chr' in chrom:
        chrom = chrom.split('chr')[1]
    # If chromosome is numeric, return as int; otherwise return a large number for sorting
    if chrom.isdigit():
        return int(chrom)
    # Assign an arbitrary large number to non-numeric chromosomes for proper sorting
    # X = 23, Y = 24, MT = 25 (for example)
    return {"X": 23, "Y": 24, "MT": 25}.get(chrom, 26)  # Default to 26 for unhandled cases

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
                        all_segments[segment][sol_name]['indep_vars'] = 0

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
                        all_segments[segment][sol_name]['indep_vars'] = 1
                else: #if number of independent variables greater than 1
                    all_segments[segment][sol_name]['indep_vars'] = len(min_max)

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
                row = {'segment_id': segment_id,
                    'chromosome': chrom,
                    'start': start,
                    'end': end,
                    'file_path': sol_name,
                    'averaged': float(np.nan),
                    'solved': False, 'fully_solved': False
                        }
            else:
                row = {'segment_id': segment_id,
                        'chromosome': chrom,
                        'start': start,
                        'end': end,
                        'file_path': sol_name}
                for t_var in sorted_t_vars:
                    y_vals = solution['vals'][1]
                    row[t_var] = np.median(y_vals[t_var]) if t_var in y_vals else np.nan
                row['averaged'] = not solution['constant']
                row['solved'] = True
                # Determine if fully solved
                row['fully_solved'] = all(
                    solution.get(t_var) == 'constant'
                    for t_var in sorted_t_vars
                ) if len(sorted_t_vars) else False
                
            rows.append(row)

    average_df = pd.DataFrame(rows)
    
    t_cols = average_df.filter(regex=r't\d').astype(float)
    normalized = t_cols.apply(lambda x: x / np.sum(x), axis=1)

    normalized_cumsum = normalized.apply(np.cumsum, axis=1)
    normalized_cumsum = normalized_cumsum.rename(columns=lambda col: f"{col}_norm_cumsum")

    normalized = normalized.rename(columns=lambda col: f"{col}_norm")
    average_df = pd.concat([average_df, normalized, normalized_cumsum], axis=1)
    average_df['major'] = average_df['file_path'].apply(lambda x: re.split(r'_|\.', os.path.basename(x))[0])
    average_df['minor'] = average_df['file_path'].apply(lambda x: re.split(r'_|\.', os.path.basename(x))[1])
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

    return all_breakpoints, all_segments, df

def get_per_segment_solutions(multiplicities_file=None, multiplicities_df=None, 
                              solutions_dir = '/n/data1/hms/dbmi/park/jbrew/Tau/downsized_solutions/',
                             signatures=["SBS1"], min_snvs = 5, min_merge_gap=0.1):
    
    if multiplicities_df is None:
        multiplicities_df = pd.read_csv(multiplicities_file, sep='\t')

    multiplicities_df['chr'] = multiplicities_df['segment_id'].apply(lambda x: parse_chromosome(x.split(':')[0]))
    multiplicities_df['chr_str'] = multiplicities_df['segment_id'].apply(lambda x:x.split(':')[0])
    multiplicities_df['start'] = multiplicities_df['segment_id'].apply(lambda x: int(re.split(':|-', x)[1]))
    multiplicities_df['end'] = multiplicities_df['segment_id'].apply(lambda x: int(re.split(':|-', x)[2]))
    multiplicities_df = multiplicities_df.sort_values(by=['chr','start'])
    mle_dict = {}
    #signatures = ["SBS1"]
    signature_tag = '_'.join(signatures)
    SBS_df = multiplicities_df.query('sig_max in @signatures')
    
    segment_solutions = defaultdict(lambda: defaultdict(dict))
    
    #new merging code
    def get_segment_info(row):
        major, minor = row[['major_cn','minor_cn']]
        cn = str(major)+'_'+str(minor)
        chrom = row['chr_str']
        start, end = row[['start', 'end']]
        length = end - start
        return cn, chrom, start, end, length
    
    i = 0
    new_df = list()
    while i < SBS_df.shape[0]-1:
        j = 1
        merged=False
        row1, row2 = SBS_df.iloc[i], SBS_df.iloc[i+j]
        cn1, chrom1, start1, end1, length1 = get_segment_info(row1)
        cn2, chrom2, start2, end2, length2 = get_segment_info(row2)
        gap = start2 - end1
        #if copy numbers are equal and gap between adjacent segments is less than 10% of smallest segment, then merge!
        merged_segments = list()
        merged_segments.append(row1['segment_id'])
        while cn1 == cn2 and chrom1 == chrom2 and gap < min_merge_gap * np.min([length1, length2]):
            merged=True
            row1 = row1.copy()
            merged_segments.append(row2['segment_id'])
            
            #updating first row to represent merged row1 and row2 
            row1['end'] = row2['end']
            row1[row1.filter(regex='N.*').index] = row1.filter(regex='N.*') + row2.filter(regex='N.*')
            row1['segment_id'] = str(chrom2) +':'+str(start1)+'-'+str(end2) 
            j+=1
            
            #updating row2 to be the NEXT row
            if i+j >= SBS_df.shape[0]-1:
                break
            row2 = SBS_df.iloc[i+j]
            cn2, chrom2, start2, end2, length2 = get_segment_info(row2)
    
        row1 = row1.copy()
        if merged:
            print(f"MERGED {len(merged_segments)} SEGMENTS: " + ','.join(merged_segments))
        row1['merged'] = merged
        new_df.append(row1)
        i+=j
    
    SBS_df = pd.DataFrame(new_df)
    #print(SBS_df)
    for idx, row in SBS_df.iterrows():
        #calculate segment by segment timing results
        segment = row['segment_id']
        major = int(row['major_cn'])
        minor = int(row['minor_cn'])
        N_vars = [f'N{i}' for i in range(1, major+1)]
        t_num = major + minor - 1 if minor > 0 else major
        if t_num == 0:
            continue
        t_vars = [f't{i}' for i in range(1, t_num+1)]
        var(N_vars)
        var(t_vars)
        N_values = {var(N_var): N_val for N_var, N_val in row.filter(items = N_vars).items()}
        if sum(N_values.values()) < min_snvs:
            continue

        max_in_row = max(int(num[1:]) for num in row.filter(regex='N.*').keys().to_list())

        #this happens if there are zero values for some high multiplicities so we have to fill in these zeros
        if major > max_in_row:
            for i in range(max_in_row+1, major+1):
                N_values[var(f'N{i}')] = 0

        if (major >= 7 and minor >= 6) or (major > 7):
            print(f"Copy number of segment ({major},{minor}) too high, skipping...")
            continue
        
        sols = get_solutions(major, minor, solutions_dir)
        unfit = defaultdict(list)
        unfit_constraints = defaultdict(list)
        
        for sol_file in sols:
            solution = load(sol_file)
            segment_solutions[segment][sol_file] = list()

            substitution = change_inequality([s.subs(N_values) for s in solution])
            no_variable = [sub for sub in substitution if len(sub.variables())==0]
            variable = [sub for sub in substitution if len(sub.variables())> 0]
            if all(no_variable):
                #split into equalities and inequalities
                equalities = [var for var in variable if var.operator() == operator.eq]
                inequalities = [var for var in variable if var.operator() in [operator.ge, operator.le]]

                #identify dependent and independent variables
                dep = [eq.lhs() for eq in equalities]
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

                curr_sol = {'constants': constants,
                        'non-constants': nonconstants,
                        'min_max': indep_min_max}
                
                segment_solutions[segment][sol_file].append(curr_sol)
            else:
                constraints = [x for x in solution if all([var(t_var) not in x.variables() for t_var in t_vars])]
                unfit[sol_file].append(solution)
                unfit_constraints[sol_file].append(constraints)
        
        ## DEALING WITH MLE estimates
        if sum([len(x) for x in segment_solutions[segment].values()]) == 0:
            #print("NO SOLUTION! Calculating MLE estimate")
            #print(segment)
            #print(segment_solutions[segment])
            #print(unfit_constraints)
            unfit_likelihoods = {}
            unfit_lambdas = {}

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

                max_likelihood = np.prod(poisson.pmf(observed_counts, mle_lambdas))
                
                unfit_likelihoods[name] = max_likelihood
                unfit_lambdas[name] = mle_lambdas

            sol_file = max(unfit_likelihoods, key=unfit_likelihoods.get)
            mle_dict[segment] = max(unfit_likelihoods.values())
            solution = load(sol_file)

            N_values = {N_val: val for N_val, val in zip(N_values.keys(), np.round(unfit_lambdas[sol_file], decimals=8))}
            substitution = change_inequality([s.subs(N_values) for s in solution])
            no_variable = [round_small_values(sub) for sub in substitution if len(sub.variables())==0]
            variable = [sub for sub in substitution if len(sub.variables())> 0]
            
            if not all(no_variable):
                print("FAILED")
                print(no_variable)
                break
            if all(no_variable):
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
                segment_solutions[segment][sol_file].append(curr_sol)

    sorted_segments = dict(sorted(
        segment_solutions.items(),
        key=lambda item: (parse_chromosome(item[0].split(':')[0]), int(item[0].split(':')[1].split('-')[0]))))
    return sorted_segments, mle_dict

def plot_timing_results(sample, all_segments, breakpoint_median, mle_dict, output_path=None, average=False, ref='hg37'):
    chr_lengths_hg38 = """chr1	248956422
chr2	242193529
chr3	198295559
chr4	190214555
chr5	181538259
chr6	170805979
chr7	159345973
chrX	156040895
chr8	145138636
chr9	138394717
chr11	135086622
chr10	133797422
chr12	133275309
chr13	114364328
chr14	107043718
chr15	101991189
chr16	90338345
chr17	83257441
chr18	80373285
chr20	64444167
chr19	58617616
chrY	57227415
chr22	50818468
chr21	46709983"""
    
    chr_lengths_hg37 = """chr1	249250621
chr2	243199373
chr3	198022430
chr4	191154276
chr5	180915260
chr6	171115067
chr7	159138663
chrX	155270560
chr8	146364022
chr9	141213431
chr10	135534747
chr11	135006516
chr12	133851895
chr13	115169878
chr14	107349540
chr15	102531392
chr16	90354753
chr17	81195210
chr18	78077248
chr20	63025520
chrY	59373566
chr19	59128983
chr22	51304566
chr21	48129895"""

    chr_lengths = chr_lengths_hg37 if ref=='hg37' else chr_lengths_hg38
    data = StringIO(chr_lengths)
    df = pd.read_csv(data, sep='\t', header=None, names=['Chromosome', 'Length'])
    chr_length_dict = dict(zip(df['Chromosome'], df['Length']))
    chr_length_dict['chr0']=0
    
    #copy_number_dict = cn_dict
    
    ### PLOTTING SEGMENT TIMINGS
    max_mle_val = max(mle_dict.values()) if len(mle_dict.values()) else 0
    fig, ax = plt.subplots(figsize=(30, 15))

    colormaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens,
                 plt.cm.Oranges, plt.cm.Purples, plt.cm.Greys,
                 plt.cm.YlGnBu, plt.cm.BuPu, plt.cm.GnBu,
                 plt.cm.PuRd, plt.cm.coolwarm, plt.cm.Spectral,
                 plt.cm.PiYG, plt.cm.BrBG, plt.cm.viridis,
                 plt.cm.plasma, plt.cm.cividis, plt.cm.magma, plt.cm.inferno]

    def get_gradated_colors(num_t_values, colormap):
        return [colormap(i / num_t_values) for i in range(1, num_t_values + 1)]

    legend_data = {cmap.name: {"handles": [], "labels": []} for cmap in colormaps}

    chrom = 'chr0'
    offset = 0
    old_segment_end = 0
    segment_xpos = -0.02
    old_segment_end = 0
    
    for segment_name, values in all_segments.items():
        line = False
        indep_vars = [x['indep_vars'] > 1 for x in values.values()]
        if all(indep_vars):
            print(">1 independent variables, skipping...")
            continue
        valid_sol_idx = np.where(['vals' in x for x in values.values()])[0]
        choice = np.random.choice(valid_sol_idx)
        choice = list(values.keys())[choice]
        segment = values[choice]  # pick most general solution (last solution!)
            
        segment_start = int(segment_name.split(':')[1].split('-')[0])
        
        new_chrom = segment_name.split(':')[0]
        if new_chrom != chrom:
            line = True
            chrom_edited = chrom if 'chr' in chrom else 'chr'+chrom
            gap = chr_length_dict[chrom_edited] - old_segment_end
            chrom = new_chrom
        else:
            gap = segment_start - old_segment_end if old_segment_end != 0 else 0
        
        #print("gap:", gap, "segment start:", segment_start, "prev segment end:", old_segment_end)
        segment_end = int(segment_name.split(':')[1].split('-')[1])
        old_segment_end = segment_end
        segment_length = segment_end - segment_start
        
        gap_x = np.linspace(0, gap, 2)+offset
        ax.fill_between(gap_x,[0]*2, [1]*2, color='white')
        offset += gap
        mle_val = mle_dict.get(segment_name, '')
        mle_val = mle_val if type(mle_val) == str else f'{mle_val:.3g}'

        if line:
            ax.vlines(offset, ymin=0, ymax=1, color='black', linestyle='--',
                     linewidth=3)
            ax.fill_between(np.linspace(offset, offset+segment_start, 2),[0]*2, [1]*2, color='white')
            offset+=segment_start
        
        if segment['vals'][0] is not None and not average:
            x_vals, y_vals = segment['vals']
            y_vals = {str(key): val for key, val in list(y_vals.items())}
            y_vals = dict(sorted(y_vals.items(), key=lambda x: int(x[0][1:])))
            cumulative_bottom = np.zeros(len(x_vals))  # Start stacking from the bottom

            # Compute the total sum at each index across all variables
            total_sum_per_index = np.sum(
                np.array([values for values in y_vals.values()], dtype=float),
                axis=0
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
                    x_vals * segment_length + offset,
                    cumulative_bottom,
                    cumulative_bottom + normalized_values,
                    color=colors[idx],
                    alpha=0.8
                )

                # Store the handle and label for the legend
                if legend_label not in legend_data[colormap.name]["labels"]:
                    legend_data[colormap.name]["handles"].append(fill)
                    legend_data[colormap.name]["labels"].append(legend_label)

                cumulative_bottom += normalized_values  # Stack for the next variable

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

                legend_label = f"t{idx + 1}/{num_t_values}"  # e.g., t1/2, t2/2, t1/3, etc.

                # Plot the segment
                fill = ax.fill_between(
                    offset + x_vals * segment_length,
                    cumulative_bottom,
                    cumulative_bottom + constant_values,
                    color=colors[idx],
                    alpha=0.8
                )

                # Store the handle and label for the legend
                if legend_label not in legend_data[colormap.name]["labels"]:
                    legend_data[colormap.name]["handles"].append(fill)
                    legend_data[colormap.name]["labels"].append(legend_label)

                cumulative_bottom += constant_values  # Stack for the next constant variable

        segment_center = offset + segment_length / 2  # Center of the segment
        ax.text(
            segment_center,
            segment_xpos,
            segment_name,
            rotation=45,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            rotation_mode='anchor'
        )
        ax.vlines(segment_center, 0, -0.015, color='black')

        major=re.split(r'_|\.', os.path.basename(choice))[0]
        minor=re.split(r'_|\.', os.path.basename(choice))[1]
        ax.text(segment_center, 1.02, str(major)+'_'+str(minor),
                fontsize=15, rotation=90,
               verticalalignment='center',horizontalalignment='center')

        #plotting MLE value
        ax.text(segment_center, 1.1, mle_val,
                fontsize=15, rotation=90,
               verticalalignment='center',horizontalalignment='center')
        offset += segment_length

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

    # Adjust the figure layout to make spaace for the legends
    plt.subplots_adjust(right=0.7)  # Increase the right margin to fit the legends
    ax.hlines(breakpoint_median, xmin = 0, xmax = offset, color='red',
              linestyle = '--', linewidth=3)
    ax.set_xlabel("")
    ax.set_ylabel("Mutation time", fontsize=20)
    average_tag = "" if not average else "_AVERAGED"
    ax.set_title(f"{sample}{average_tag}", fontsize=30, y=1.1)
    ax.grid(True)
    plt.ylim(-0.1999,1.05)
    ax.set_xticks([])
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()

def calculate_timing_solutions(sample, multiplicities_file=None, 
                               multiplicities_df=None, output_tsv = None, 
                               output_plot=None, average=False,
                               solutions_dir = '/n/data1/hms/dbmi/park/jbrew/Tau/downsized_solutions/',
                              ref='hg37', plot=True, min_snvs = 5, min_merge_gap=0.01):
        #solutions_dir = '/n/data1/hms/dbmi/park/jbrew/matrices/new_solutions'):

    sorted_segments, mle_dict = get_per_segment_solutions(multiplicities_file=multiplicities_file, 
                                                                  multiplicities_df=multiplicities_df, 
                                                                  solutions_dir=solutions_dir, 
                                                          min_snvs=min_snvs, min_merge_gap=min_merge_gap)
    #solutions_df = process_solutions(sorted_segments)
    breakpoints, all_segments, solutions_df = calculate_breakpoints(sorted_segments)
    
    solutions_df['seg_length'] = solutions_df['end'].astype(int) - solutions_df['start'].astype(int)
    seg_lengths = np.array(solutions_df['seg_length'])
    breakpoints = [np.array([x for x in y if x > 1e-2 and x < 9.9e-1]) for y in solutions_df['breakpoints']]
    weighted_median_vals = [np.repeat(x, math.ceil(y*1e-7)) for x,y in zip(breakpoints, seg_lengths)]
    weighted_median = np.median([x for y in weighted_median_vals for x in y])
    if output_tsv:
        solutions_df.to_csv(output_tsv)
    if output_plot:
        plot_timing_results(sample, all_segments, breakpoint_median=weighted_median, 
                            output_path=output_plot, average=average, mle_dict=mle_dict, ref=ref)
    elif plot:
        plot_timing_results(sample, all_segments, breakpoint_median=weighted_median, average=average, mle_dict=mle_dict, ref=ref)
    else:
        return sorted_segments, solutions_df, breakpoints
        
    return sorted_segments, solutions_df, breakpoints
