import argparse
import logging
from sympy import sympify, solve

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate solutions from the solution file against multiplicities.")
    parser.add_argument("--multiplicities", required=True, help="Path to the input multiplicities file.")
    parser.add_argument("--solutions", required=True, help="Path to the solution file.")
    parser.add_argument("--output", required=True, help="Path to save the evaluated solutions.")
    return parser.parse_args()

def parse_conditions(condition_string):
    return [sympify(cond) for cond in condition_string.strip('[]').split(', ')]

def process_solution(segment_id, sig_max, major_cn, minor_cn, conditions, context):
    results = []
    for condition in conditions:
        try:
            evaluated = solve(condition, dict=True, **context)
            results.append(f"{condition}: {evaluated}")
        except Exception as e:
            logging.error(f"Error evaluating condition: {condition}. Context: {context}. Error: {e}")
            results.append(f"{condition}: Error ({e})")
    return results

def main():
    args = parse_arguments()
    
    # Read input files
    with open(args.multiplicities, 'r') as multiplicities_file:
        multiplicities_lines = multiplicities_file.readlines()
    
    with open(args.solutions, 'r') as solutions_file:
        solutions_lines = solutions_file.readlines()
    
    # Parse multiplicities into a dictionary
    multiplicities = {}
    for line in multiplicities_lines[1:]:  # Skip header
        fields = line.strip().split('\t')
        segment_id, sig_max, major_cn, minor_cn = fields[:4]
        context = {f"N{i+1}": int(fields[4+i]) for i in range(len(fields) - 4)}
        multiplicities[(segment_id, sig_max)] = (major_cn, minor_cn, context)
    
    # Evaluate solutions and write results
    with open(args.output, 'w') as output_file:
        output_file.write("segment_id,sig_max,major_cn,minor_cn,conditions_and_results,context\n")
        
        for line in solutions_lines:
            if not line.strip():
                continue  # Skip empty lines
            
            fields = line.strip().split('\t')
            if len(fields) != 2:
                logging.warning(f"Skipping malformed line: {line.strip()}")
                continue
            
            solution_segment, conditions_string = fields
            conditions = parse_conditions(conditions_string)
            
            # Extract segment_id and sig_max
            if '_' not in solution_segment or not solution_segment.endswith('_solution.txt'):
                logging.warning(f"Skipping malformed solution segment: {solution_segment}")
                continue
            
            segment_id, sig_max = solution_segment.replace('_solution.txt', '').split('_', 1)
            
            # Retrieve multiplicities context
            key = (segment_id, sig_max)
            if key not in multiplicities:
                logging.warning(f"Missing multiplicities for key: {key}")
                continue
            
            major_cn, minor_cn, context = multiplicities[key]
            
            logging.debug(f"Processing segment_id: {segment_id}, sig_max: {sig_max}, "
                          f"major_cn: {major_cn}, minor_cn: {minor_cn}")
            logging.debug(f"Context: {context}")
            logging.debug(f"Conditions: {conditions}")
            
            # Evaluate conditions
            results = process_solution(segment_id, sig_max, major_cn, minor_cn, conditions, context)
            results_str = "; ".join(results)
            
            # Write the results to the output file
            output_file.write(
                f"{segment_id},{sig_max},{major_cn},{minor_cn},{results_str},{context}\n"
            )

if __name__ == "__main__":
    main()
