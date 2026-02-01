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
import operator
from scipy.stats import mannwhitneyu, binomtest
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from io import StringIO
from scipy.stats import poisson, chi2

#general helper functions
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
def round_small_values(expr, tol=1e-2):
    if expr.is_relational():  # If the expression is an inequality or equation
        return expr.operator()(round_small_values(expr.lhs(), tol), round_small_values(expr.rhs(), tol))
    elif expr.operator() is None:  # If the expression is a number or variable
        return 0 if abs(expr) < tol else expr
    else:  
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

def null_model_log_likelihood(observed_counts):
    log_L = 0
    for N_i, count in observed_counts.items():
        log_L += poisson.logpmf(count, count)  # lambda = N_i
    return log_L

def alt_model_log_likelihood(observed_counts, lambdas):
    log_L = 0
    for N_i, count in observed_counts.items():
        log_L += poisson.logpmf(count, lambdas[N_i])
    return log_L

def likelihood_ratio_test(log_L_null, log_L_alt, df):
    likelihood_ratio = -2 * (log_L_null - log_L_alt)
    p_value = chi2.sf(likelihood_ratio, df)
    return likelihood_ratio, p_value
    
def decompose_solution(variable, t_vars):
    #split into equalities and inequalities
    equalities = [var for var in variable if var.operator() == operator.eq]
    inequalities = [var for var in variable if var.operator() in [operator.ge, operator.le]]

    #identify dependent and independent variables
    dep = [eq.lhs() for eq in equalities]
    indep = list(set(var(t_vars)) - set(dep))

    equalities = {var.lhs(): var.rhs() for var in equalities}

    #split equalities into constants and non-constants
    constants = {var: round_small_values(value) for var, value in equalities.items() if len(value.variables()) == 0}
    nonconstants = {var: round_small_values(value) for var, value in equalities.items() if len(value.variables()) > 0}

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
    return constants, nonconstants, indep_min_max

def merge_segments(SBS_df, max_gap=1000, max_seg_size=1e5, by_state=False):
    def get_segment_info(row):
        major, minor = row[['major_cn', 'minor_cn']]
        cn = f"{major}_{minor}"
        chrom = row['chr_str']
        start, end = row[['start', 'end']]
        length = end - start
        return cn, chrom, start, end, length

    new_df = []
    if 'sample' not in SBS_df.columns:
        SBS_df['sample'] = 1

    if by_state:
        grouped = SBS_df.groupby(['chr_str', 'major_cn', 'minor_cn','sample'])
        for (chrom, major, minor, sample), group in grouped:
            start = 0 #group['start'].min()
            end = sum(group['end'] - group['start'])
            merged_row = group.iloc[0].copy()
            merged_row['start'] = 0
            merged_row['end'] = end
            merged_row['segment_id'] = f"{chrom}:{start}-{end}"
            merged_row['merged'] = True
            merged_row['sample'] = sample
            N_cols = group.filter(regex=r'^N\d+$').columns
            merged_row[N_cols] = group[N_cols].sum()
            new_df.append(merged_row)
        return pd.DataFrame(new_df)

    # Default merge behavior
    i = 0
    while i < SBS_df.shape[0] - 1:
        j = 1
        merged = False
        row1, row2 = SBS_df.iloc[i], SBS_df.iloc[i + j]
        cn1, chrom1, start1, end1, length1 = get_segment_info(row1)
        cn2, chrom2, start2, end2, length2 = get_segment_info(row2)
        gap = start2 - end1
        merged_segments = [row1['segment_id']]

        while (
            cn1 == cn2 and
            chrom1 == chrom2 and
            gap <= max_gap and
            length1 <= max_seg_size and
            length2 <= max_seg_size
        ):
            merged = True
            row1 = row1.copy()
            merged_segments.append(row2['segment_id'])

            # Update merged row
            row1['end'] = row2['end']
            row1[row1.filter(regex='N.*').index] = row1.filter(regex='N.*') + row2.filter(regex='N.*')
            row1['segment_id'] = f"{chrom2}:{start1}-{end2}"

            j += 1
            if i + j >= SBS_df.shape[0]:
                break

            row2 = SBS_df.iloc[i + j]
            cn2, chrom2, start2, end2, length2 = get_segment_info(row2)
            gap = start2 - row1['end']
            length1 = row1['end'] - start1

        if merged:
            print(f"MERGED {len(merged_segments)} SEGMENTS: " + ', '.join(merged_segments))
            row1['merged'] = True
        else:
            row1 = row1.copy()
            row1['merged'] = False

        new_df.append(row1)
        i += j

    return pd.DataFrame(new_df)

def calculate_solution_metrics(segment_id, solution, sol_file, N_values, copy_number,
                               exclusion_reason='', constraints_satisfied=True,
                              MLE_solution=False, MLE_values={}):
    
    chrom, positions = segment_id.split(':')
    start, end = positions.split('-')[0], positions.split('-')[1]
    plotting_data = solution.get('plotting', None) if solution else None
    metrics_data = solution.get('metrics', None) if solution else None
    
    metrics = {'segment_id': segment_id,
            'chromosome': chrom,
            'start': start,
            'end': end,
            'major': copy_number[0],
            'minor': copy_number[1],
            'segment length': float(end) - float(start) + 1,
            'solution_path': sol_file if sol_file is not None else float(np.nan),
            'independent_variables': None if metrics_data is None else plotting_data.get('indep_vars',''),
            'segment_exclusion_reason': exclusion_reason,
            'MLE_solution': MLE_solution,
            'constraints_satisfied': constraints_satisfied}

    if len(exclusion_reason) > 0 or (not metrics['constraints_satisfied'] and 
                                                                     not metrics['MLE_solution']):
        return metrics

    sorted_N_values = sorted(N_values, key=lambda x: int(str(x)[1:]))
    N_val_string = ''
    total = 0
    for N_var, N_val in N_values.items():
        N_val_string += str(N_var) + '=' + str(N_val) + ';'
        total+=int(N_val)
    metrics['multiplicity_counts'] = N_val_string
    metrics['total_count'] = total
    
    if len(MLE_values):
        total = 0
        MLE_sorted_N_values = sorted(N_values, key=lambda x: int(str(x)[1:]))
        MLE_N_val_string = ''
        for N_var, N_val in MLE_values.items():
            MLE_N_val_string += str(N_var) + '=' + str(N_val) + ';'
            total += int(N_val)
        metrics['MLE_counts'] = MLE_N_val_string
        metrics['total_MLE_count'] = total
    
    pattern = r"t\d+"
    #print(plotting_data)
    if plotting_data['indep_vars'] > 1:
        return metrics
        
    all_t_vars = [x for x in plotting_data['vals'][1].keys() if re.match(pattern, str(x))]
    sorted_t_vars = sorted(all_t_vars, key=lambda x: int(str(x)[1:])) if all_t_vars else []
    #print(sorted_t_vars)
    for t_var in sorted_t_vars:
        #print(t_var)
        y_vals = plotting_data['vals'][1]
        if not isinstance(y_vals[t_var], np.ndarray):
            var_start = var_end = y_vals[t_var]
        else:
            var_start, var_end = y_vals[t_var][0], y_vals[t_var][2]
        metrics[str(t_var)] = var_start if var_start == var_end else (var_start, var_end)
    
    return metrics    

def compute_mle_solution(segment, solution, N_values, N_vars, t_vars, major):
    constraints_for_N = [s for s in solution if all(var(t) not in s.variables() for t in t_vars)]
    #constraints_for_N = change_inequality(constraints_for_N)
    eqs = [c for c in constraints_for_N if c.operator() == operator.eq]
    ineqs = [c for c in constraints_for_N if c.operator() != operator.eq]
    #print(f'eqs: {eqs}')
    #print(f'ineqs: {ineqs}')
    ineq_coeffs = extract_coefficients_from_constraints(ineqs, var(N_vars))
    eq_coeffs = extract_coefficients_from_constraints(eqs, var(N_vars))
    #print(f'ineq_coeffs: {ineq_coeffs}')
    #print(f'eq_coeffs: {eq_coeffs}')
    
    cons = []
    for c in ineq_coeffs:
        cons.append({'type': 'ineq', 'fun': lambda x, c=c: np.dot(c, x)})
    for c in eq_coeffs:
        cons.append({'type': 'eq', 'fun': lambda x, c=c: np.dot(c, x)})

    obs_counts = list(N_values.values())
    guess = np.full(len(obs_counts), np.mean(obs_counts))
    result = minimize(neg_log_likelihood, guess, args=(obs_counts,), constraints=cons)
    mle_lambdas = result.x
    mle_lambdas_dict = {var(f'N{i+1}'): float(val) for i, val in enumerate(mle_lambdas)}
    
    log_L_alt = alt_model_log_likelihood(N_values, mle_lambdas_dict)
    log_L_null = null_model_log_likelihood(N_values)
    rel_like = np.exp(log_L_alt - log_L_null)

    if rel_like < 0.5:
        #print(f'[{segment}] relative likelihood too low! skipping solving for t-values')
        t_struct = {'constants': {}, 'non-constants': {}, 'min_max': {}}
        mle_metrics = {
        'MLE_solution': True,
        'mle_lambdas': mle_lambdas_dict,
        'log_L_alt': log_L_alt,
        'relative_likelihood': rel_like,
        'raw_L_alt': np.exp(log_L_alt)
        }
        mle_metrics.update(t_struct)
        return mle_metrics
    
    sub_mle = [s.subs(mle_lambdas_dict) for s in solution]
    no_vars = [s for s in sub_mle if len(s.variables()) == 0]
    vars_left = [s for s in sub_mle if len(s.variables()) > 0]
    if all(no_vars):
        constants, nonconstants, indep_min_max = decompose_solution(vars_left, t_vars)
        t_struct = {'constants': constants, 'non-constants': nonconstants, 'min_max': indep_min_max}
    else:
        #print(f"[{segment}] FLOATING POINT ERROR! Rounding conditions")
        #print(f"[{segment}] non-rounded conditions: {no_vars}")
        rounded_no_vars = [round_small_values(x) for x in no_vars]
        #print(f"[{segment}] rounded conditions: {rounded_no_vars}")
        if all(rounded_no_vars):
            #print(f"[{segment}] rounding succeeded!")
            constants, nonconstants, indep_min_max = decompose_solution(vars_left, t_vars)
            t_struct = {'constants': constants, 'non-constants': nonconstants, 'min_max': indep_min_max}
        else:
            #print("FAILED EVEN AFTER ROUND! CHECK CODE. CONTINUING ANYWAY...")
            constants, nonconstants, indep_min_max = decompose_solution(vars_left, t_vars)
            t_struct = {'constants': constants, 'non-constants': nonconstants, 'min_max': indep_min_max}
    
    mle_metrics = {
        'MLE_solution': True,
        'mle_lambdas': mle_lambdas_dict,
        'log_L_alt': log_L_alt,
        'relative_likelihood': rel_like,
        'raw_L_alt': np.exp(log_L_alt)
    }
    mle_metrics.update(t_struct)
    return mle_metrics    

#(doesn't have code for dealing with high error samples)
def process_candidate_solution(solution, N_values, N_vars, t_vars, segment, copy_number, sol_file=""):
    # Record the original observed N values.
    orig_N_values = N_values.copy()
    MLE_solution_flag = False
    sol_name = os.path.basename(sol_file).split('_solution')[0]
    solution_updated_inequality = change_inequality(solution)
    #print(f'solution: {solution_updated_inequality}')
    # Substitute N_values into each candidate equation.
    substitution = [s.subs(N_values) for s in solution_updated_inequality]
    #print(f'substitution: {substitution}')
    no_variable = [sub for sub in substitution if len(sub.variables()) == 0]
    #print(f'substitution constraints: {no_variable}')
    constraints_satisfied = all(no_variable)
    
    # If constraints are not satisfied with observed values, immediately compute and apply MLE.
    if not constraints_satisfied:
        #print(f"[{segment},{sol_name}] CONSTRAINTS NOT SATISFIED, TRYING MLE")
        mle_metrics = compute_mle_solution(segment, solution_updated_inequality, N_values, N_vars, t_vars, copy_number[0])
        #update N_values based on MLE
        #print(f"[{segment},{sol_name}] Original N values: {N_values}")
        N_values = mle_metrics.get('mle_lambdas', N_values)
        #print(f"[{segment},{sol_name}] MLE N values: {N_values}")
        #print(f"[{segment},{sol_name}] Likelihood ratio: {mle_metrics.get('relative_likelihood', None)}")
        MLE_solution_flag = True

    rel_like = None
    # For a solved candidate, use decompose_solution on 'variable'
    if constraints_satisfied:
        variable = [sub for sub in substitution if len(sub.variables()) > 0]
        constants, nonconstants, indep_min_max = decompose_solution(variable, t_vars)
    elif MLE_solution_flag:
        constants = mle_metrics.get('constants','')
        nonconstants = mle_metrics.get('non-constants','')
        indep_min_max = mle_metrics.get('min_max',{})
        rel_like = mle_metrics.get('relative_likelihood', '')
    else:
        constants, nonconstants, indep_min_max = {}, {}, {}

    sol_struct = {
        'constants': constants,
        'non-constants': nonconstants,
        'min_max': indep_min_max,
        'MLE_solution': MLE_solution_flag,
        'orig_N_values': orig_N_values,
        'mle_N_values': N_values if MLE_solution_flag else {},
        'indep_vars': len(indep_min_max.keys()),
        'relative_likelihood': rel_like
    }

    # Compute segment-level metrics.
    chrom, positions = segment.split(':')
    start, end = positions.split('-')[0], positions.split('-')[1]
    metrics = {
        'segment_id': segment,
        'chromosome': chrom,
        'start': start,
        'end': end,
        'major': copy_number[0],
        'minor': copy_number[1],
        'solution_name': os.path.basename(sol_file).split('_solution')[0],
        'segment length': float(end) - float(start) + 1,
        'solution_path': sol_file if sol_file is not None else np.nan,
        'independent_variables': sol_struct.get('indep_vars'),
        'segment_exclusion_reason': '',
        'MLE_solution': MLE_solution_flag,
        'constraints_satisfied': constraints_satisfied
    }
    
    # Aggregate the N values.
    N_val_string, total = "", 0
    for N_var, N_val in orig_N_values.items():
        N_val_string += f"{N_var}={N_val};"
        total += int(N_val)
    metrics['multiplicity_counts'] = N_val_string
    metrics['total_count'] = total

    MLE_N_val_string = ""
    if MLE_solution_flag:
        for N_var, N_val in N_values.items():
            MLE_N_val_string += f"{N_var}={N_val};"
    metrics['MLE_multiplicity_counts'] = MLE_N_val_string
    
    # If there is one or less independent variable, add the t variable values.
    #if sol_struct.get('indep_vars', 0) <= 1:
    r'''
        if indep_min_max:
            indep_t_var, (lower, upper) = list(indep_min_max.items())[0]
        else:
            indep_t_var = None
        pattern = r"t\d+"
        sorted_t_vars = sorted(t_vars, key=lambda x: int(str(x)[1:])) if t_vars else []
        constant_list = [str(x) for x in constants.keys()]
        nonconstant_list = [str(x) for x in nonconstants.keys()]
        for t_var in sorted_t_vars:
            if str(t_var) == str(indep_t_var):
                val = (lower, upper)
            elif str(t_var) in constant_list:
                val = constants[var(t_var)]
            elif str(t_var) in nonconstant_list:
                expr = nonconstants[var(t_var)]
                left_val, right_val = expr.subs({indep_t_var:lower}), expr.subs({indep_t_var:upper})
                val = left_val if left_val == right_val else (left_val, right_val)
            else:
                print('failed')
                print(metrics)
            metrics[str(t_var)] = val
    '''
    sol_struct.update(metrics)
    return sol_struct

def get_per_segment_solutions(multiplicities_file=None, multiplicities_df=None,
                              solutions_dir='/n/data1/hms/dbmi/park/jbrew/Tau/downsized_solutions/',
                              signatures=["SBS1", "SBS5"], min_snvs=5, merge=False,
                              by_state=False, max_seg_size=50000, chromosomes=None):
    # Load multiplicities data.
    if multiplicities_df is None:
        multiplicities_df = pd.read_csv(multiplicities_file, sep='\t')

    # Parse segment metadata.
    multiplicities_df['chr'] = multiplicities_df['segment_id'].apply(lambda x: parse_chromosome(x.split(':')[0]))
    multiplicities_df['chr_str'] = multiplicities_df['segment_id'].apply(lambda x: x.split(':')[0])
    multiplicities_df['start'] = multiplicities_df['segment_id'].apply(lambda x: int(re.split(':|-', x)[1]))
    multiplicities_df['end'] = multiplicities_df['segment_id'].apply(lambda x: int(re.split(':|-', x)[2]))
    multiplicities_df = multiplicities_df.sort_values(by=['chr', 'start'])
    #if 'sample' in multiplicities_df.columns:
    #    multiplicities_df['segment_id'] = multiplicities_df['segment_id'] + '-' + multiplicities_df['sample']

    if 'sample' not in multiplicities_df.columns:
        multiplicities_df['sample'] = ''

    if chromosomes is None:
        chromosomes = list(multiplicities_df.chr_str.unique())

    # Filter by signatures.
    SBS_df = multiplicities_df.query('sig_max in @signatures and chr_str in @chromosomes')
    agg_dict = {col: ('sum' if col.startswith('N') else 'first')
            for col in SBS_df.columns if col not in ('segment_id','sample')}
    SBS_df = SBS_df.groupby(['segment_id','sample']).agg(agg_dict).reset_index()
    if 'sig_max' in SBS_df.columns:
        SBS_df.drop(columns=['sig_max'], inplace=True)

    SBS_df = SBS_df.sort_values(by=['chr', 'start'])
    #print(SBS_df.head())
    #print(SBS_df.shape)
    segment_solutions = defaultdict(lambda: defaultdict(dict))
    if merge:
        SBS_df = merge_segments(SBS_df, by_state=by_state, max_seg_size = max_seg_size)

    if 'sample' in SBS_df.columns:
        SBS_df['segment_id'] = SBS_df['segment_id'] + '-' + SBS_df['sample']
    
    # Process each segment.
    for idx, row in SBS_df.iterrows():
        segment = row['segment_id']
        major = int(row['major_cn'])
        minor = int(row['minor_cn'])
        N_vars = [f'N{i}' for i in range(1, major + 1)]
        t_num = (major + minor - 1) if minor > 0 else major
        if t_num == 0:
            continue
        t_vars = [f't{i}' for i in range(1, t_num + 1)]
        
        var(N_vars)
        var(t_vars)

        #now also fills in gaps if we have missing N variable that isn't present in the table!
        N_values = {var(N_var): row[N_var] if N_var in row.index else 0 for N_var in N_vars}

        if sum(N_values.values()) < min_snvs:
            exclusion = f'Fewer than {min_snvs} total SNVs'
            print(f"[{segment}]: {exclusion}, skipping...")
            segment_solutions[segment]['excluded_segment_metrics'] = calculate_solution_metrics(
                segment, None, None, N_values, (major, minor),
                exclusion_reason=exclusion, constraints_satisfied=False)
            continue
        
        if (major >= 7 and minor >= 6) or (major > 7):
            print(f"[{segment}]: Copy number of segment ({major},{minor}) too high, skipping...")
            exclusion = f'Copy number too high (limit is {major}_{minor})'
            segment_solutions[segment]['excluded_segment_metrics'] = calculate_solution_metrics(
                segment, None, None, N_values, (major, minor),
                exclusion_reason=exclusion, constraints_satisfied=False)
            continue
        
        sols = get_solutions(major, minor, solutions_dir)
        num_sols = len(sols)
        print(f'[{segment}], {major}_{minor}, processing {num_sols} solutions...')
        for i, sol_file in enumerate(sols):
            solution_obj = load(sol_file)
            sol_metrics = process_candidate_solution(solution_obj, N_values, N_vars, t_vars,
                                                     segment, (major, minor), sol_file)
            sol_name = os.path.basename(sol_file).split('_solution')[0]
            segment_solutions[segment][sol_name] = sol_metrics
            if (i+1) % 50 == 0:
                print(f'[{segment}], {major}_{minor}, processed {i+1} solutions so far, {len(sols) - (i+1)} left to go')
                
        print(f'[{segment}], {major}_{minor}, complete!')
    sorted_segments = dict(sorted(segment_solutions.items(), key=lambda item: ('-'.join(item[0].split('-')[2:]), 
                                                    parse_chromosome(item[0].split(':')[0]),
                                                    int(item[0].split(':')[1].split('-')[0]))))
    return sorted_segments

def plot_timing_results(sample, solutions, events={}, output_path=None, clustered=True, ref='hg37', seg_to_sol_dict=None, gain_dict=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    from io import StringIO
    from matplotlib.lines import Line2D  # For custom legend handles

    # Chromosome lengths definitions.
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
    df = pd.read_csv(data, sep='\t', header=None, names=['Chromosome','Length'])
    chr_length_dict = dict(zip(df['Chromosome'], df['Length']))
    chr_length_dict['chr0'] = 0

    # Create figure.
    fig, ax = plt.subplots(figsize=(30, 15))
    colormaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens,
                 plt.cm.Oranges, plt.cm.Purples, plt.cm.Greys,
                 plt.cm.YlGnBu, plt.cm.BuPu, plt.cm.GnBu,
                 plt.cm.PuRd, plt.cm.coolwarm, plt.cm.Spectral,
                 plt.cm.PiYG, plt.cm.BrBG, plt.cm.viridis,
                 plt.cm.plasma, plt.cm.cividis, plt.cm.magma, plt.cm.inferno]

    # Cluster colors for gain events.
    cluster_colors = {
        'G0': 'blue',
        'G1': 'green',
        'G2': 'purple',
        'G3': 'orange',
        'G4': 'cyan',
        'flat': 'gray'
    }
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_yticks([])

    def get_gradated_colors(n, cmap):
        return [cmap(i/n) for i in range(1, n+1)]
    
    # Legend dictionary for candidate time segments (colormap-based).
    legend_data = {cmap.name: {"handles": [], "labels": []} for cmap in colormaps}
    
    segments_sorted = solutions
    global_offset = 0

    current_chrom = 0
    last_seg_end = 0
    
    # Set to track clusters encountered in gain_dict.
    clusters_found = set()

    for segment_name in segments_sorted:
        cand = solutions[segment_name]
        if seg_to_sol_dict is None:
            candidate = np.random.choice(list(cand.values()))
        elif segment_name not in seg_to_sol_dict:
            continue
        else:
            candidate = cand.get(seg_to_sol_dict[segment_name])
            if candidate is None:
                continue

        # Parse segment details.
        sample = ''
        try:
            seg_chrom = segment_name.split(':')[0]
            rest = segment_name.split(':')[1].split('-')
            if len(rest) > 2:
                seg_coords = '-'.join(rest[:2])
                sample = ';' + '-'.join(rest[2:])
            else:
                seg_coords = rest
            seg_start, seg_end = map(int, seg_coords.split('-'))
        except Exception:
            continue
        seg_length = seg_end - seg_start

        # Check if starting a new chromosome.
        if seg_chrom != current_chrom:
            chrom_length = chr_length_dict.get('chr' + str(current_chrom), 0)
            if last_seg_end < chrom_length:
                gap = chrom_length - last_seg_end
                gap_x = np.linspace(0, gap, 2) + global_offset
                ax.fill_between(gap_x, 0, 1, color='white')
                global_offset += gap
            current_chrom = seg_chrom
            chrom_length = chr_length_dict.get('chr' + str(current_chrom), 0)
            last_seg_end = seg_start
            ax.text(global_offset + chrom_length/2, -0.05, current_chrom, fontsize=12, ha="left", va="bottom")
            ax.vlines(global_offset, 0, 1, color='black', linestyle='--', linewidth=2)
            if seg_start > 0:
                gap = seg_start
                gap_x = np.linspace(0, gap, 2) + global_offset
                ax.fill_between(gap_x, 0, 1, color='white')
                global_offset += gap
        
        if last_seg_end is not None and seg_start > last_seg_end:
            gap = seg_start - last_seg_end
            gap_x = np.linspace(0, gap, 2) + global_offset
            ax.fill_between(gap_x, 0, 1, color='white')
            global_offset += gap
        
        seg_global_start = global_offset
        seg_global_end = global_offset + seg_length


        #---------------- BELOW NEEDS EDITING ------------------------------------------------------------
        t_vals = dict(_extract_t_series(candidate))
        t_values = list(t_vals.values())

        left_vals = np.array([x[0] if isinstance(x, tuple) else x for x in t_values])
        right_vals = np.array([x[1] if isinstance(x, tuple) else x for x in t_values])
        left_val_sum = np.sum(left_vals)
        right_val_sum = np.sum(right_vals)

        norm_left = np.array(left_vals / left_val_sum, dtype=float)
        norm_right = np.array(right_vals / right_val_sum, dtype=float)
        norm = np.column_stack((norm_left, norm_right))

        x_vals = np.linspace(0, seg_length, 2)
        cumulative = np.array([0, 0], dtype=float)
        num_t = len(t_values)
        cmap = colormaps[(num_t-1) % len(colormaps)]
        colors = get_gradated_colors(num_t, cmap)
        for idx, norm_vals in enumerate(norm):
            fill = ax.fill_between(x_vals + global_offset, cumulative, cumulative + norm_vals,
                                   color=colors[idx], alpha=0.8)
            if gain_dict:
                gain_key = f'{segment_name}_t{idx+1}'
                cluster, norm_time = gain_dict.get(gain_key, (None, None))
                if cluster:
                    clusters_found.add(cluster)
                    ax.hlines(norm_time, xmin=global_offset, xmax=global_offset + seg_length, 
                              color=cluster_colors.get(cluster, 'black'), linewidth=4)
            label = f"t{idx+1}/{num_t}"
            if label not in legend_data[cmap.name]["labels"]:
                legend_data[cmap.name]["handles"].append(fill)
                legend_data[cmap.name]["labels"].append(label)
            cumulative += norm_vals

        #---------------- ABOVE NEEDS EDITING ------------------------------------------------------------
        
        seg_center = global_offset + seg_length/2
        
        sol_name = candidate.get('solution_name', '')
        lr_val = candidate.get('relative_likelihood', '')
        if lr_val not in ["", None] and not isinstance(lr_val, str):
            lr_val = f"; LR: {lr_val:.2f}"
        else:
            lr_val = ""
        total_count = candidate.get('total_count', '')
        
        if seg_length > 0.07 * chrom_length:
            ax.text(seg_center, 1.01, sol_name + '; ' + str(total_count) + lr_val + sample,
                    fontsize=10, rotation=60, verticalalignment='center',
                    horizontalalignment='left', rotation_mode='anchor')
        
        global_offset += seg_length
        last_seg_end = seg_end

    if current_chrom is not None:
        chrom_length = chr_length_dict.get(current_chrom, 0)
        if last_seg_end < chrom_length:
            gap = chrom_length - last_seg_end
            gap_x = np.linspace(0, gap, 2) + global_offset
            ax.fill_between(gap_x, 0, 1, color='white')
            global_offset += gap
            ax.vlines(global_offset, 0, 1, color='black', linestyle='--', linewidth=2)
            ax.text(global_offset - chrom_length/2, -0.05, current_chrom,
                    fontsize=12, ha="center", va="bottom")
    
    # Draw legends for candidate time segments (colormaps).
    legend_spacing = 0.23
    for cmap in colormaps:
        if cmap.name in legend_data and legend_data[cmap.name]["handles"]:
            handles = legend_data[cmap.name]["handles"]
            labels = legend_data[cmap.name]["labels"]
            legend = ax.legend(handles, labels, title=f"{cmap.name} (n={len(labels)})",
                               loc="center left", fontsize=10, title_fontsize=10,
                               bbox_to_anchor=(0.955, 1.2 - legend_spacing), frameon=False)
            legend_spacing *= 1.4
            ax.add_artist(legend)
    
    # Add a separate legend for cluster assignments if any were found.
    if gain_dict and clusters_found:
        cluster_handles = [Line2D([0], [0], color=cluster_colors.get(cluster, 'black'), lw=4)
                           for cluster in sorted(clusters_found)]
        cluster_labels = sorted(clusters_found)
        # Position the cluster legend below the chromosome names (near the bottom of the plot).
        cluster_legend = ax.legend(cluster_handles, cluster_labels, title='Cluster Assignment', 
                                   loc='upper center', bbox_to_anchor=(0.5, 0.1), 
                                   ncol=len(cluster_handles), fontsize=12, title_fontsize=12, frameon=True)
        ax.add_artist(cluster_legend)
    
    # Adjust subplot margins: leave room at the bottom for the cluster legend.
    plt.subplots_adjust(bottom=0.25, right=0.75)

    if clustered:
        for clust, gain_event in events.items():
            ax.hlines(gain_event, xmin=0, xmax=global_offset, color=cluster_colors.get(clust, 'black'), linestyle='--', linewidth=3)
    elif len(events):
        for event in events:
            ax.hlines(event, xmin=0, xmax=global_offset, color='red', linestyle='--', linewidth=3)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized mutation time", fontsize=20)
    ax.set_title(f"{sample}", fontsize=30, y=1.1)
    ax.grid(True)
    plt.ylim(-0.2, 1.05)
    ax.set_xticks([])
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()

def _extract_t_series(row):
    maj  = int(row['major'])
    minr = int(row['minor'])
    
    # how many t‑variables a (major,minor) segment has
    num_t = maj + minr - (1 if minr > 0 else 0)
    t_names = [f't{i+1}' for i in range(num_t)]
    
    const_dict    = row['constants']       if isinstance(row['constants'], dict)       else {}
    nonconst_dict = row['non-constants']   if isinstance(row['non-constants'], dict)   else {}
    min_max       = row['min_max']         if isinstance(row['min_max'], dict)         else {}
    
    if min_max:
        indep_t, (lo, hi) = next(iter(min_max.items()))
        indep_t_str = str(indep_t)
    else:
        indep_t_str, lo, hi = None, None, None
    
    vals = []
    for tn in t_names:
        if tn == indep_t_str:
            vals.append((float(lo), float(hi)))
        elif tn in map(str, const_dict.keys()):
            vals.append(float(const_dict[var(tn)]))
        elif tn in map(str, nonconst_dict.keys()):
            expr   = nonconst_dict[var(tn)]
            left   = float(expr.subs({indep_t: lo}))
            right  = float(expr.subs({indep_t: hi}))
            vals.append((left, right) if left != right else left)
        else:
            vals.append(0.0)          # not defined ⇒ 0
    return pd.Series(vals, index=t_names)

def parse_mutation_counts(count_str):
    """Convert 'N1=4;N2=3;' into total count (4 + 3 = 7)"""
    if pd.isna(count_str):
        return 0
    return sum(int(part.split('=')[1]) for part in count_str.split(';') if '=' in part)
    
def candidate_priority(candidate, tol=1e-2, lr_threshold=0.5):
    zeros = set()
    t_var_series = _extract_t_series(candidate)
    #max_t_var = t_var_series.index[-1]
    t_vars = dict(t_var_series)
    # Loop over candidate keys that appear to be t variables.
    for key, val in t_vars.items():
        if isinstance(val, tuple):
            num_val = min(val)  # choose the minimum of the tuple
        else:
            num_val = float(val)
        if abs(num_val) < tol:
            # Extract the numeric part of the variable name (e.g. "t2" becomes 2)
            t_index = int(re.findall(r"\d+", str(key))[0])
            zeros.add(t_index)

    # Retrieve diff_allele_indices.
    diff = candidate.get('diff_allele_indices', [])
    if isinstance(diff, np.ndarray):
        try:
            diff = set(int(x) for x in diff.tolist())
        except Exception:
            diff = set()
    elif isinstance(diff, str):
        if diff.strip() == "":
            diff = set()
        else:
            diff = set(int(x.strip()) for x in diff.split(',') if x.strip().isdigit())
    else:
        diff = set()
    
    # Check the likelihood ratio.
    lr = candidate.get('relative_likelihood', None)
    try:
        lr_value = float(lr)
    except Exception:
        lr_value = None
    if lr_value is not None and lr_value < lr_threshold:
        return -1  # Negative priority if likelihood ratio is below threshold.
    
    return 1 if zeros.issubset(diff) else 0

def downsample(group, seed=42):
    group = group.copy()
    group['priority'] = group.apply(lambda row: candidate_priority(row), axis=1)
    # Prioritize candidates with priority == 1.
    prioritized = group[group['priority'] == 1]
    neutral = group[group['priority'] == 0]
    penalized = group[group['priority'] == -1]
    if len(prioritized) > 0:
        return prioritized.sample(n=1, random_state=seed)
    elif len(neutral) > 0:
        # If none meet the positive criteria, sample randomly.
        return neutral.sample(n=1, random_state=seed)
    else:
        return None

def visualize_breakpoints(sample, solutions_df, seed=42, output_plot=None, thresholds = [30,50]):

    valid_df = solutions_df.query('(MLE_solution or constraints_satisfied) and independent_variables < 2')
    double_solutions = solutions_df.groupby('segment_id').size().where(lambda x: x > 1).dropna().index
    solutions_df['solution'] = solutions_df['solution_path'].apply(lambda x: os.path.basename(x).split('_solutions')[0] if not pd.isnull(x) else x)
    unique_segments_df = valid_df.groupby('segment_id').apply(downsample, seed=seed, include_groups=False).reset_index()
    
    df = unique_segments_df.copy()

    determined = df.query('not MLE_solution and independent_variables == 0')
    undetermined = df.query('not MLE_solution and independent_variables > 0')
    mle = df.query('MLE_solution')
    
    # Label systems
    df["system_type"] = "other"
    df.loc[determined.index, "system_type"] = "determined"
    df.loc[undetermined.index, "system_type"] = "undetermined"
    df.loc[mle.index, "system_type"] = "mle"
    
    breakpoints_data = []
    for idx, row in df.iterrows():
        #t_vars = row.filter(regex=r't\d+$').dropna()
        #max_t_var = 't'+ str(max([int(x.split('t')[1].split('_')[0]) for x in t_vars.index]))
        t_var_series = _extract_t_series(row)
        max_t_var = t_var_series.index[-1]
        t_vars = dict(t_var_series)
        
        #if last t value is tuple, make sure to pick nonzero value!
        if isinstance(t_vars[max_t_var], tuple):
            choice = int(bool(t_vars[max_t_var][0] == 0))
        else:
            choice = np.random.choice(2)
        cumulative_time = 0
        for t_var, val in t_vars.items():
            time = val[choice] if isinstance(val, tuple) else val
            prev_cumulative_time = float(cumulative_time)
            cumulative_time += time
            is_duplicate = False
            likelihood = row["relative_likelihood"] if not pd.isnull(row["relative_likelihood"]) else 1
            if np.isclose(float(cumulative_time), prev_cumulative_time, atol=1e-02):
                is_duplicate = True
            breakpoints_data.append({
                "segment_id": row["segment_id"],
                "length": row["segment length"],
                "chromosome": row["chromosome"],
                "position": (int(row["start"]) + int(row["end"]))/2,
                "system_type": row["system_type"],
                "weight": 2/3*(row["total_count"]/df["total_count"].max()) + #weighting for total mutation count
                         1/3*likelihood, #weighting for MLE
                "time": time,
                "cumulative_time": cumulative_time,
                "up_to_t_var": t_var,
                "terminal_t_var": max_t_var,
                "copy_number": str(row["major"])+'_'+str(row["minor"]),
                "is_duplicate": is_duplicate,
                "total_counts": row['total_count'],
                "likelihood": row['relative_likelihood']
            })
    
    breakpoints_df = pd.DataFrame(breakpoints_data)
    breakpoints_df['total_time'] = breakpoints_df.groupby('segment_id')['time'].transform('sum')
    breakpoints_df['norm_time'] =  breakpoints_df['cumulative_time']/breakpoints_df['total_time']
    
    final_breakpoints = breakpoints_df.query('up_to_t_var != terminal_t_var')

    for lab in ['determined','undetermined', 'mle']:
        plt.hist(final_breakpoints.query('system_type==@lab and not is_duplicate')['norm_time'].astype(float), 
                 density=False, label=lab, alpha=0.5)
    plt.legend()
    if output_plot:
        plot_dir = os.path.dirname(output_plot)
        plt.savefig(os.path.join(plot_dir, f'{sample}_histogram1.pdf'))
        
    plt.show()
    plt.close()
    
    plt.hist(final_breakpoints.query('system_type != "mle" and not is_duplicate')['norm_time'].astype(float), 
                 density=False, label='not MLE', alpha=0.5)
    plt.hist(final_breakpoints.query('system_type == "mle" and not is_duplicate')['norm_time'].astype(float), 
                 density=False, label='MLE', alpha=0.5)
    plt.legend()
    if output_plot:
        plt.savefig(os.path.join(plot_dir, f'{sample}_histogram2.pdf'))
    plt.show()
    plt.close()

    low = thresholds[0]
    high = thresholds[1]
    condition = 'not is_duplicate'
    low_count = f'total_counts < {low}'
    mid_count = f'total_counts >= {low}'
    high_count = f'total_counts > {high}'
    plt.hist(final_breakpoints.query(f'{condition} and {low_count}')['norm_time'].astype(float), 
                 density=False, label=f'{low_count}', alpha=0.5)
    plt.hist(final_breakpoints.query(f'{condition} and {mid_count} and not {high_count}')['norm_time'].astype(float), 
                 density=False, label=f'{mid_count} and <= {high}', alpha=0.5)
    plt.hist(final_breakpoints.query(f'{condition} and {high_count}')['norm_time'].astype(float), 
                 density=False, label=f'{high_count}', alpha=0.5)
    plt.legend()
    if output_plot:
        plt.savefig(os.path.join(plot_dir, f'{sample}_histogram3.pdf'))
    plt.show()

    return final_breakpoints

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

def cluster_breakpoints(sample, breakpoints, flat_weight=0.1, output_plot=None, output_tsv=None, max_clusters=3):
    # If flat_weight is a scalar, create a list from it.
    if not isinstance(flat_weight, (list, tuple, np.ndarray)):
        flat_weight_values = [flat_weight]
    else:
        flat_weight_values = list(flat_weight)
    
    # Filter breakpoints and make an explicit copy.
    filtered_breakpoints = breakpoints.query('total_counts > 40 and not is_duplicate').copy()
    times = filtered_breakpoints['norm_time'].astype(float)
    times = np.array(times)
    
    def evaluate_gmm_with_flat(n_gaussians, data, variance_threshold=0.05, flat_weight=0.1, bic=True):
        """
        Fit a GMM with n_gaussians and evaluate an adjusted score including a flat component.
        Returns an adjusted score (lower is better) and the fitted GMM.
        """
        X = data.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_gaussians, random_state=42)
        gmm.fit(X)
        
        # Compute density of the Gaussian mixture for the data points.
        gmm_density = np.exp(gmm.score_samples(X))
        
        # Uniform density over [0,1] is 1.
        combined_density = (1 - flat_weight) * gmm_density + flat_weight * 1.0
        
        # Compute the overall log likelihood under the combined model.
        log_likelihood = np.sum(np.log(combined_density))
        
        # Calculate a BIC penalty OR .
        n_params = n_gaussians * 2
        N = len(data)
        x = np.log(N) if bic else 2
        score = -2 * log_likelihood + n_params * x #np.log(N)
        
        # penalty for overly broad Gaussian components.
        penalty = 0
        for var in gmm.covariances_.flatten():
            if var > variance_threshold:
                penalty += (var - variance_threshold) * 100  # tuning factor
        
        total_score = score + penalty
        return total_score, gmm
    
    # Grid search: try candidate flat_weight values and 1 to 3 Gaussian components.
    grid_results = []
    for candidate in flat_weight_values:
        for n in range(1, max_clusters+1):  # Trying between 1 and 3 Gaussian components by default
            score, model = evaluate_gmm_with_flat(n, times, variance_threshold=0.05, flat_weight=candidate)
            grid_results.append((n, candidate, score, model))
            print(f"Flat weight: {candidate}, Gaussian Components: {n}, Total Score: {score}")
    
    # Choose the best combination (lowest score).
    best_n, best_flat_weight, best_score, best_model = min(grid_results, key=lambda x: x[2])
    print(f"Best model: {best_n} Gaussian components with flat_weight = {best_flat_weight} (score = {best_score}).")
    
    # Assign each datapoint to a cluster.
    X = times.reshape(-1, 1)
    N = X.shape[0]
    k = best_model.means_.shape[0]  # Number of Gaussian components
    
    # Calculate density for each Gaussian component at X.
    gaussian_densities = np.zeros((N, k))
    for j in range(k):
        mean = best_model.means_[j, 0]
        # Covariance is stored as (1,1) per component; extract the scalar variance.
        variance = best_model.covariances_[j, 0, 0]
        gaussian_densities[:, j] = (1 - best_flat_weight) * best_model.weights_[j] * \
                                   norm.pdf(X.flatten(), loc=mean, scale=np.sqrt(variance))
    
    # Flat component density (uniform over [0,1]) weighted by best_flat_weight.
    flat_density = np.full(N, best_flat_weight)
    
    # Total density for each datapoint.
    total_density = flat_density + np.sum(gaussian_densities, axis=1)
    
    # Calculate densities normalized per datapoint.
    norm_densities_gaussians = gaussian_densities / total_density[:, None]  # shape (N, k)
    norm_densities_flat = flat_density / total_density  # shape (N,)
    combined_norm_densities = np.hstack([norm_densities_gaussians, norm_densities_flat[:, None]])
    
    # Assign each datapoint to the cluster with the normalized responsibility
    assignments = np.argmax(combined_norm_densities, axis=1)
    
    # Create labels: Gaussian clusters ("G0", "G1", ...) and the flat cluster ("flat").
    cluster_labels = ['G{}'.format(i) for i in range(k)] + ['flat']
    assigned_labels = [cluster_labels[i] for i in assignments]
    
    # Update the DataFrame with cluster assignments.
    filtered_breakpoints.loc[:, 'cluster_idx'] = assignments
    filtered_breakpoints.loc[:, 'cluster_label'] = assigned_labels
    
    # Report counts per cluster.
    cluster_counts = filtered_breakpoints['cluster_label'].value_counts()
    print("\nDatapoint counts per cluster:")
    print(cluster_counts)

    # Map cluster label to corresponding Gaussian mean.
    cluster_mean_map = {f'G{j}': best_model.means_[j, 0] for j in range(k)}
    cluster_mean_map['flat'] = np.nan  # flat cluster has no defined mean

    # Assign the cluster mean as a new column.
    filtered_breakpoints.loc[:, 'cluster_mean'] = filtered_breakpoints['cluster_label'].map(cluster_mean_map)

    print(filtered_breakpoints.head())
    # --------------------------
    # Create a figure with two subplots side by side.
    # --------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    x = np.linspace(0, 1, 1000)
    
    # Define color mapping for clusters.
    colors = {
        'G0': 'blue',
        'G1': 'green',
        'G2': 'purple',
        'G3': 'orange',
        'G4': 'cyan',
        'flat': 'gray'
    }
    
    # --------------------------
    # Plot 1: Segregated by cluster with individual Gaussian densities overlaid.
    # --------------------------
    for j in range(k):
        mean_val = best_model.means_[j, 0]
        variance = best_model.covariances_[j, 0, 0]
        weight_j = best_model.weights_[j]
        density_j = (1 - best_flat_weight) * weight_j * norm.pdf(x, loc=mean_val, scale=np.sqrt(variance))
        # Use .get() with default of 1 so that if a cluster has zero points, it does not error.
        count_j = cluster_counts.get(f'G{j}', 1)
        scaling = np.log(count_j * 100)  # adjust scaling factor as needed.
        ax1.plot(x, density_j * scaling, color=colors.get(f'G{j}', None), lw=2, 
                 label=f"Gaussian G{j} (μ={mean_val:.3f})")
    
    # Plot histograms for each cluster on ax1.
    for clust in np.unique(assigned_labels):
        idx = filtered_breakpoints['cluster_label'] == clust
        cluster_data = filtered_breakpoints.loc[idx, 'norm_time'].astype(float)
        n = len(cluster_data)
        bins = int(np.ceil(np.log(n))) if n > 1 else 20
        ax1.hist(cluster_data, bins=bins, alpha=0.5,
                 label=f"{clust} (n={n})", color=colors.get(clust, None))
    
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Frequency")
    ax1.set_xlim(0, 1)
    ax1.set_title("Gain Timings Colored by Gaussian Cluster Assignments")
    ax1.legend(fontsize=8)
    
    # --------------------------
    # Plot 2: Overall histogram with individual Gaussian densities overlaid.
    # --------------------------
    overall_bins = 40
    ax2.hist(times, bins=overall_bins, alpha=0.5, label="All data", color="lightgray")
    
    bin_width = (1 - 0) / overall_bins  # bin width for overall data
    scale_factor = len(times) * bin_width  # scale factor to convert density to counts
    
    for j in range(k):
        mean_val = best_model.means_[j, 0]
        variance = best_model.covariances_[j, 0, 0]
        weight_j = best_model.weights_[j]
        density_j = (1 - best_flat_weight) * weight_j * norm.pdf(x, loc=mean_val, scale=np.sqrt(variance))
        density_j_scaled = density_j * scale_factor
        ax2.plot(x, density_j_scaled, color=colors.get(f'G{j}', None), lw=2, 
                 label=f"Gaussian G{j} (μ={mean_val:.3f})")
    
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Frequency")
    ax2.set_xlim(0, 1)
    ax2.set_title("All Gain Timings with Gaussians Overlaid")
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    # If the flag output_plot is True, save the figure.
    if output_plot:
        output_plot_dir = os.path.dirname(output_plot)
        output_fig_path = os.path.join(output_plot_dir, f'{sample}_clustering_results.pdf')
        plt.savefig(output_fig_path)
    
    plt.show()
    
    if output_tsv:
        output_tsv_dir = os.path.dirname(output_tsv)
        output_tsv_path = os.path.join(output_tsv_dir, f'{sample}_clustering_results.tsv')
        filtered_breakpoints.to_csv(output_tsv_path, sep='\t', index=False)
    
    return filtered_breakpoints, best_model

def calculate_timing_solutions(sample, multiplicities_file=None, 
                               multiplicities_df=None, output_tsv = None, 
                               output_plot=None, cluster=True,
                               solutions_dir = '/n/data1/hms/dbmi/park/jbrew/Tau/downsized_solutions/',
                              ref='hg37', plot=True, min_snvs = 5, 
                               merge=False, by_state=False, max_seg_size = 50000, 
                               sigs=["SBS1"], chromosomes = None):
        #solutions_dir = '/n/data1/hms/dbmi/park/jbrew/matrices/new_solutions'):

    sorted_segments = get_per_segment_solutions(multiplicities_file=multiplicities_file, 
                                                                  multiplicities_df=multiplicities_df, 
                                                                  solutions_dir=solutions_dir, signatures=sigs,
                                                          min_snvs=min_snvs, merge=merge, max_seg_size=max_seg_size, by_state=by_state,
                                               chromosomes = chromosomes)
    
    rows=[]
    for segment, data in sorted_segments.items():
        if 'excluded_segment_metrics' in data:
            rows.append(data['excluded_segment_metrics'])
        else:
            for sol_file, sol in data.items():
                rows.append(sol)
    
    solutions_df = pd.DataFrame(rows)
    if 'solution_name' not in solutions_df.columns:
        solutions_df['solution_name'] = (
        solutions_df['solution_path']
        .fillna('')
        .apply(lambda p: os.path.basename(p).split('_solution')[0] if p else '')
        )
    
    matrix_df = pd.read_csv('/n/data1/hms/dbmi/park/jbrew/Tau/new_matrix_code/counts_diagram_As.txt',sep='\t')
    unique_indices = [np.unique(re.split(',|;', x)) if isinstance(x, str) else np.nan for x in matrix_df['diff_allele_indices'].tolist()]
    matrix_df['diff_allele_unique'] = unique_indices
    tag_to_diff_alleles = dict(zip(matrix_df['tag_A'], matrix_df['diff_allele_unique']))
    solutions_df['diff_allele_indices'] = [tag_to_diff_alleles.get(x,'') for x in solutions_df['solution_name']]
    
    #NEW METHOD OF PICKING SOLUTIONS FROM SEGMENTS 
    #(prioritizes t-value zeroes which imply simultaneous gains on different alleles)
    valid_df = solutions_df.query('(MLE_solution or constraints_satisfied) and independent_variables < 2') 
    unique_segments_df = valid_df.groupby('segment_id').apply(downsample, include_groups=False).reset_index()
    segment_to_solution_dict = dict(zip(unique_segments_df['segment_id'], unique_segments_df['solution_name']))

    selected_keys = set(zip(unique_segments_df['segment_id'], unique_segments_df['solution_name']))
    solutions_df['selected'] = solutions_df.apply(
        lambda row: (row['segment_id'], row['solution_name']) in selected_keys, axis=1
    )
    
    #plot_dir = os.path.dirname(output_plot) if output_plot else ''
    
    all_breakpoints = visualize_breakpoints(sample, solutions_df, output_plot=output_plot)
    weighted_median = np.median(all_breakpoints.query('not is_duplicate')['norm_time'])

    if all_breakpoints.query('not is_duplicate').shape[0] < 5:
        cluster = False
        
    if cluster:
        gain_events_df, model = cluster_breakpoints(sample, all_breakpoints, flat_weight=np.linspace(0, 0.4, 41), 
                                                    output_plot=output_plot, output_tsv=output_tsv)
        gain_events_df.loc[:, 'unique_id'] = gain_events_df.segment_id + '_' + gain_events_df.up_to_t_var
        gain_dictionary = dict(zip(gain_events_df.unique_id, zip(gain_events_df.cluster_label, gain_events_df.norm_time)))
    else:
        gain_dictionary = None
    
    events = {f'G{i}': mean for i, mean in enumerate(model.means_)} if cluster else [weighted_median]
    
    if output_tsv:
        solutions_df.to_csv(output_tsv, sep='\t')
    if output_plot:
        plot_timing_results(sample, sorted_segments, events=events, 
                            output_path=output_plot, ref=ref, seg_to_sol_dict = segment_to_solution_dict, 
                            gain_dict = gain_dictionary, clustered=cluster)
    elif plot:
        plot_timing_results(sample, sorted_segments, events=events, ref=ref,
                           seg_to_sol_dict = segment_to_solution_dict, gain_dict=gain_dictionary, clustered=cluster)
    else:
        return sorted_segments, solutions_df, all_breakpoints
        
    return sorted_segments, solutions_df, all_breakpoints
