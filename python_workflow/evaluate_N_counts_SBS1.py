#!/usr/bin/env python3
#SBATCH -p short
#SBATCH -A park
#SBATCH -t 30:00
#SBATCH --mem=5G
#SBATCH -e timing_solutions_SBS1_logs/%j.err
#SBATCH -o timing_solutions_SBS1_logs/%j.out

from sage.all import load, var, solve
import sys
import re
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

multiplicities_file = sys.argv[1]
output_name = os.path.basename(multiplicities_file).split('_multiplicities.txt')[0]

multiplicities_df = pd.read_csv(multiplicities_file, sep='\t')
directory = 'timing_plots_SBS1'
copy_number_dict = dict(zip(multiplicities_df['segment_id'], 
                       multiplicities_df['major_cn'].astype(str)+'_' + multiplicities_df['minor_cn'].astype(str)))

signatures = ["SBS1"]
signature_tag = '_'.join(signatures)
SBS_df = multiplicities_df.query('sig_max in @signatures')

if SBS_df.shape[0] == 0:
    print("NO SBS1 IN SAMPLE!")
    with open(f'{directory}/no_SBS1/{output_name}_no_SBS1.txt', 'w') as fp:
        pass
    exit(0)

solutions_dir = '/n/data1/hms/dbmi/park/jbrew/matrices/new_solutions'
all_solutions = os.listdir(solutions_dir)

def get_solutions(major, minor):
    #get only solution files compatible with copy number state 
    pattern = f"{major}_{minor}.*_solutions.sobj"
    sols = [solutions_dir + '/' + re.match(pattern, sol).group() for sol in all_solutions if re.match(pattern, sol) is not None]
    return sols

#def process_solution(sol_file, N_vars, t_vars, N_values):
    #eqs, ineqs, solutions = load(sol_file)
    #var(N_vars)
    #var(t_vars)

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

    if sum(N_values.values()) < 8:
        continue

    max_in_row = max(int(num[1:]) for num in row.filter(regex='N.*').keys().to_list())

    #this happens if there are zero values for some high multiplicities so we have to fill in these zeros
    if major > max_in_row:
        for i in range(max_in_row+1, major+1):
            N_values[var(f'N{i}')] = 0

    sols = get_solutions(major, minor) 
    #print("N_VALUES:", N_values)

    #collect solutions that don't fulfil inequalities of solutions 
    unfit = defaultdict(list) 
    fit = defaultdict(list)

    #solution_found = False

    for sol_file in sols:
        #if solution_found:
        #    break
        #print('SOLUTIONS FILE:', sol_file)
        #load solutions
        eqs, ineqs, solutions = load(sol_file)
        sol_num = len(solutions)
        segment_solutions[segment][sol_file] = list()

        for i, solution in enumerate(solutions):
            substitution = [s.subs(N_values) for s in solution] 
            no_variable = [sub for sub in substitution if len(sub.variables())==0]
            variable = [sub for sub in substitution if len(sub.variables())> 0]
            if all(no_variable):
                #solution_found = True
                #split into equalities and inequalities
                equalities = [var for var in variable if str(var.operator()) == '<built-in function eq>']
                inequalities = [var for var in variable if str(var.operator()) in ['<built-in function lt>', '<built-in function gt>']]
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
                    operators = [str(ineq.operator()) for ineq in inequalities if ind in ineq.variables()]
                    indep_min = float('inf')
                    indep_max = -float('inf')
                    for ineq, operator in zip(relevant_ineqs, operators):
                        if not len(ineq.lhs().variables()):
                            if operator == '<built-in function lt>':
                                indep_min = np.minimum(indep_min, ineq.lhs())
                            else:
                                indep_max = np.maximum(indep_max, ineq.lhs())
                        else:
                            if operator == '<built-in function gt>':
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
                unfit[sol_file].append(solution)
    #if len(fit) == 0:
       ##print(f"NO SOLUTIONS FOUND FOR {row['segment_id']}! copy number state was {major}_{minor} and N values were {N_values}")   

output_name = os.path.basename(multiplicities_file).split('_multiplicities.txt')[0]
#with open(f'timing_solutions/{output_name}_solutions.pkl', 'wb') as f: 
#    pickle.dump(segment_solutions, f)

#print({seg: sum([len(sol) for sol_file, sol in sols.items()]) for seg, sols in segment_solutions.items()})
#print(segment_solutions)

#sort the segments
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

#SAVING SOLUTIONS TO TSV

all_segments = defaultdict(list)

for segment, data in sorted_segments.items():
    #print(segment)
    for sample, solutions in data.items():
        for solution in solutions:
            #print(solution)
            min_max = solution.get("min_max", {})
            #print(min_max)
            if len(min_max) <= 1:
                if len(min_max) == 1:  # Only process single independent variable segments
                    var, bounds = list(min_max.items())[0]
                    lower, upper = bounds
                    x_vals = np.linspace(float(lower), float(upper), 100) 

                    non_constants = solution.get("non-constants", {})
                    y_vals = {}
                    for dep_variable, dep_value in non_constants.items():
                        y_vals[dep_variable] = np.array([float(dep_value.subs({var: val})) for val in x_vals])

                    y_vals[var] = x_vals

                    all_segments[segment].append((x_vals, y_vals))

                if len(min_max) == 0:
                    constants = solution.get("constants", {})

                    y_vals = constants

                    all_segments[segment].append((None, y_vals))
                
from sage.all import load, var, solve
def process_data(data):
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
                        'file_path': file_path,
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
    standard_cols = ['chromosome', 'start', 'end', 'file_path']
    sum_cols = [f'sum_t1_to_t{k}' for k in range(1, max_t+1)] if max_t > 0 else []
    return pd.DataFrame(rows)[standard_cols + sorted_t_vars + sum_cols + ['fully_solved']]

# Usage:
df = process_data(segment_solutions)
df = df.rename(columns={variable: str(variable) for variable in df.filter(regex=r'^t\d+').columns})
df['tree_structure'] = [os.path.basename(x).split('_solution')[0] for x in df['file_path']]
del df['file_path']
if len(df.filter(regex=r'\bt\d+').columns) == 1:
    directory = 'timing_plots_SBS1/no_gains'
df.to_csv(f'{directory}/{output_name}_solutions.tsv', sep='\t')
df_full = df

### CALCULATING BREAKPOINTS

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest
import matplotlib.pyplot as plt
import re
from scipy.spatial import KDTree

# Filter for fully solved segments
df = df.query('fully_solved')
# Extract timing and sum columns
t_columns = [col for col in df.columns if re.match(r'^t\d+$', col)]
sum_columns = [col for col in df.columns if re.match(r'^sum_t1_to_t\d+$', col)]

# Normalize timing variables to sum to 1 for each segment
df.loc[:, sum_columns] = (df.loc[:, sum_columns].div(df.loc[:, t_columns].sum(axis=1), axis=0)).astype(float)
df.loc[:, t_columns] = (df.loc[:, t_columns].div(df.loc[:, t_columns].sum(axis=1), axis=0)).astype(float)

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

print('breakpoints:', all_breakpoints)
median_breakpoint = np.median(all_breakpoints)


### PLOTTING SEGMENT TIMINGS
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(30, 15))

# Define a list of colormaps to use for gradations
colormaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples, plt.cm.Greys,plt.cm.YlGnBu, plt.cm.BuPu, plt.cm.GnBu, plt.cm.PuRd, plt.cm.coolwarm, plt.cm.Spectral, plt.cm.PiYG, plt.cm.BrBG, plt.cm.viridis, plt.cm.plasma, plt.cm.cividis, plt.cm.magma, plt.cm.inferno]

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
    #choice = np.random.choice(len(values))
    segment = values[-1]  # pick most general solution (last solution!)
    segment_start = int(segment_name.split(':')[1].split('-')[0])
    segment_end = int(segment_name.split(':')[1].split('-')[1])
    new_chrom = segment_name.split(':')[0]
    
    if new_chrom != chrom:
        line = True
        chrom = new_chrom
    
    if line:
        ax.vlines(offset, ymin=0, ymax=1, color='black', linestyle='--',
                 linewidth=3)

    if segment[0] is not None:  # For segments with `x_vals`
        x_vals, y_vals = segment
        y_vals = {str(key): val for key, val in list(y_vals.items())}
        y_vals = dict(sorted(y_vals.items(), key=lambda x: int(x[0][1:])))
        cumulative_bottom = np.zeros(len(x_vals))  # Start stacking from the bottom

        # Compute the total sum at each index across all variables
        total_sum_per_index = np.sum(
            np.array([values for values in y_vals.values()], dtype=float),
            axis=0,
        )
        x_vals = np.linspace(0, 1, 100)

        # Determine the number of t values and select the appropriate colormap
        num_t_values = len(y_vals)
        colormap = colormaps[(num_t_values - 1) % len(colormaps)]  # Cycle through colormaps
        colors = get_gradated_colors(num_t_values, colormap)

        for idx, (variable, values) in enumerate(y_vals.items()):
            numeric_values = np.array(values, dtype=float)  # Ensure numeric
            normalized_values = numeric_values / total_sum_per_index

            # Generate the legend label dynamically
            if num_t_values == 1:
                legend_label = "1/1"  # Special case for 1 t value
            else:
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

    elif segment[1] is not None:  # For constants
        y_vals = segment[1]
        y_vals = {str(key): val for key, val in list(y_vals.items())}
        y_vals = dict(sorted(y_vals.items(), key=lambda x: int(x[0][1:])))
        x_vals = np.linspace(0, 1, 100)
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
        legend_spacing = legend_spacing * 1.28
        ax.add_artist(legend)  # Add the legend to the plot

# Adjust the figure layout to make space for the legends
plt.subplots_adjust(right=0.7)  # Increase the right margin to fit the legends
ax.hlines(median_breakpoint, xmin = 0, xmax = offset, color='red', 
          linestyle = '--', linewidth=3)
ax.set_xlabel("")
ax.set_ylabel("Mutation time", fontsize=20)
ax.set_title(f"{output_name}", fontsize=30)
ax.grid(True)
plt.ylim(-0.1999,1.05)
ax.set_xticks([])
plt.tight_layout()
plt.savefig(f"{directory}/{output_name}_segment_plot_{signature_tag}.png", dpi=300)
plt.close()

### NEAREST NEIGHBOR DISTANCES

from scipy.stats import ranksums

# Step 1: Define a clustering metric (nearest neighbor distance)
def compute_nearest_neighbor_distance(breakpoints):
    # Use KDTree for efficient nearest neighbor search
    breakpoints = np.array(breakpoints).reshape(-1, 1)  # Reshape to 2D for KDTree
    tree = KDTree(breakpoints)
    distances, _ = tree.query(breakpoints, k=2)  # k=2 because the closest point is itself
    return distances[:, 1]  # Return the mean distance to the nearest neighbor

# Step 2: Simulate null distribution
def simulate_null_breakpoints(df, num_simulations=1):
    null_nn_distances = []
    for _ in range(num_simulations):
        simulated_breakpoints = []
        for _, row in df.iterrows():
            # Get the number of breakpoints in this segment
            num_breakpoints = len([x for x in row['breakpoints'] if x not in [0,1]])
            simulated_segment_breakpoints = np.sort(np.random.uniform(0, 1, num_breakpoints))
            simulated_breakpoints.extend(simulated_segment_breakpoints)
        null_nn_distances.extend(compute_nearest_neighbor_distance(simulated_breakpoints))
        #print(null_nn_distances)
    return null_nn_distances
# Compute observed nearest neighbor distance of all pooled breakpoints

if not np.isnan(all_breakpoints).all() and len(all_breakpoints) > 1:
    observed_nn_distance = compute_nearest_neighbor_distance(all_breakpoints)
    null_nn_distances = simulate_null_breakpoints(df)
else:
    observed_nn_distance = np.nan
    null_nn_distances = np.nan

p_value = ranksums(observed_nn_distance, null_nn_distances)[1]
print(f"p-value: {p_value}")
solutions_per_segment = {segment : np.sum([len(solutions) > 0 for name, solutions in x.items()]) for segment, x in segment_solutions.items()}

segment_valid_solution_proportions = {segment : (sum([len(sol) > 0 for sol_name, sol in x.items()]), len(x)) for segment, x in segment_solutions.items()}

per_segment_stats = pd.DataFrame({'segment':list(segment_valid_solution_proportions.keys()),
                                 'valid_solutions' : list(segment_valid_solution_proportions.values()),
                                'fraction_valid': [x[0]/x[1] for x in segment_valid_solution_proportions.values()],
                                'copy_number' : [copy_number_dict[x] for x in segment_valid_solution_proportions.keys()]})

per_segment_stats.to_csv(f'{directory}/{output_name}_per_segment_stats.tsv', sep='\t')

if len([x[1] for x in per_segment_stats['valid_solutions'] if x[1] > 1]):
    valid_sols = sum([x[0] for x in per_segment_stats['valid_solutions'] if x[1] > 1])
    total_sols = sum([x[1] for x in per_segment_stats['valid_solutions'] if x[1] > 1])
    total_fraction_valid = valid_sols/total_sols
else:
    total_fraction_valid = np.nan

segment_valid_solution_proportions = {segment : (sum([len(sol) > 0 for sol_name, sol in x.items()]), len(x)) for segment, x in segment_solutions.items()}

summary_stats_df = pd.DataFrame({'sample': [f'{output_name}'], 'p_value':[p_value], 'observed_mean_nn_distance':[np.mean(observed_nn_distance)],
             'null_mean_nn_distance':[np.mean(observed_nn_distance)],
             'median_number_of_solutions': np.median(list(solutions_per_segment.values())),
              'mean_number_of_solutions': np.mean(list(solutions_per_segment.values())),
             'number_of_segments_with_no_solution': sum([x == 0 for x in solutions_per_segment.values()]),
             'number_of_segments_with_one_solution': sum([x == 1 for x in solutions_per_segment.values()]),
            'number_of_segments_with_multiple_solutions': sum([x > 1 for x in solutions_per_segment.values()]),
            'solution_number_distribution': str(dict(zip(np.unique(list(solutions_per_segment.values()), return_counts=True)[0],
np.unique(list(solutions_per_segment.values()), return_counts=True)[1]))),
             'total_fraction_valid': total_fraction_valid}) 

summary_stats_df.to_csv(f'{directory}/{output_name}_summary_stats.tsv', sep='\t')
    
import matplotlib.pyplot as plt
import numpy as np

# Check if observed_nn_distance is NaN
if np.isnan(observed_nn_distance).all() or np.isnan(null_nn_distances).all():
    with open(f'{directory}/{output_name}_not_enough_breakpoints.txt', 'w') as fp:
        pass
else:
    # Calculate p-value
    p_value = ranksums(observed_nn_distance, null_nn_distances)[1]
    print(f"p-value: {p_value}")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot null distribution and retrieve density values
    null_counts, null_bins, _ = plt.hist(null_nn_distances, bins=50, alpha=0.7, color='blue', label='Null Distribution', density=True)

    # Plot tumor sample distribution and retrieve density values
    observed_counts, observed_bins, _ = plt.hist(observed_nn_distance, color='red', density=True, bins=50, alpha=0.7, label='Tumor Sample')

    # Calculate maximum density from both distributions and add padding
    max_density = max(null_counts.max(), observed_counts.max())
    ymax = max_density * 1.1  # 10% padding above the highest bar

    # Add vertical lines with dynamic ymax
    plt.ylim(0, ymax)
    plt.vlines(np.mean(null_nn_distances), ymin=0, ymax=ymax, linestyle='--', color='blue', label='Null Mean')
    plt.vlines(np.mean(observed_nn_distance), ymin=0, ymax=ymax, linestyle='--', color='red', label='Tumor Mean')
    plt.text(max(observed_nn_distance) / 4, max_density * 1.05, f'p=%.2e' % p_value, horizontalalignment='center')

    # Add labels, title, and legend
    plt.xlabel('Nearest Neighbor Distance')
    plt.ylabel('Density')
    plt.title(f'{output_name}: Tumor Sample vs Null Distribution of NN Distance')
    plt.legend()
    plt.show()

    plt.hist(all_breakpoints, bins=20)
    plt.title(f'{output_name}: Copy Number Gain times')
    plt.xlabel('time')
    plt.savefig(f"{directory}/{output_name}_gain_times_{signature_tag}.png", dpi=300)
    plt.close()
