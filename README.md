# Timing analysis with Tau

- [Background](#background)
	- [Diagram representation](#diagram-representation)
	- [Matrix representation of diagrams](#matrix-representation-of-diagrams)
	- [Systematically compose the A matrices](#systematically-compose-the-A-matrices)
	- [Solving for time when there is no exact solution](#solving-for-time-when-there-is-no-exact-solution)

# Background

<img align="right" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Mutations_as_clocks_at_amplified_segments.png" width="400">

Point mutations can be used to time copy number amplifications based on:
* the assumption that mutation counts reflect molecular clock, meaning older cells carry more mutations.
* the simple observation that the early amplifications will carry late point mutations, meaning they were generated after the amplification, on single copies of the chromosome. While, late amplifications will carry early mutations, generated before the amplifications, that are duplicated on multiple chromosomes.

## Diagram representation

Here is a diagram showing the time evolution for a diploid chromosome segment (on the left hand side) that evolves into a major copy number 3 and minor copy number 2 state.

![Image description](https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Diagram_to_equations.png)

Assuming molecular time is measured by number of mutations, the time intervals are expected to be proportional to number of mutations based on the equations listed above (add the t intervals for each color). 

### Single state can be achieved in multiple ways

For the example above we have three diagrams that differ from each other based on when minor allele is amplified with respect to major allele.

![Image description](https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/diagrams.png)

Two of the diagrams results in the same set of equations, while third one where amplification in minor segments happen first differs.

### Amplifications lead to more DNA material leading to a higher rate of mutations than diploid chromosome

Despite the lack of exact solutions for several copy number evolution trajectories, it is always possible to apply a normalization to the mutation counts such that the boost on the number of mutations due to more DNA material can be cancelled out. This normalization has a geometric interpretation as demonstrated for the M = 3, N = 2 state in the diagram below by the dashed lines that complete the missing lines in our grid. The green lines are duplicated (meaning a factor of 2 for N2) and the red line tripled (meaning a factor of 3 for N3). N<sub>1</sub> + 2N<sub>2</sub> + 3N<sub>3</sub> 

![Image description](https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Normalization.png)

## Matrix representation of diagrams

The set of linear equations can be represented as a matrix equation, for the example above:

<img class="center" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Matrix_representation.png" width=500>

In the A matrix, the placement of values in each row indicates final CN values, for example a value of 1 in row 3 indicates that the segment will be amplified to 3 copies, a value of 2 in row 2 indicates two segments that will be amplified to 2 copies, a diagonal change in the numbers mean a branching from a given state to another, see the arrows below. In addition, matrix A can be expressed as the sum of contributions of minor (N) and major (M) chromosome, which are composed of branching (B<sub>M</sub> and B<sub>N</sub>) and propagation matrices (P<sub>M</sub> and P<sub>N</sub>)

<img class="center" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/factorizing_A_matrix.png" width = 700>

For the above matrices the graphical represantation of propagation and branching matrices are as shown below:

<img class="center" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/propagations_branchings.png" width="600">

## Systematically compose the A matrices

The decomposition into branching and propagation matrices makes the calculation of A matrices easier.

*Branching matrices:* Given an initial copy number state a propagation matrix is supposed to take all the copy numbers from the bottom to the top rows while moving from left to right such that the next column will have a change in higher rows, say of index i, of magnitude -1, and this will be compensated by the increase of + 1in rows that represent the branching products in the following column. In the example above for the major copy number, the first column contains a value of 1 at the row 3, meaning that this allele will eventually generate 3 copies, which branches into a 2 and a 1, which means these branches will later lead to 2 and 1 copies.

*Propagation matrices*: are constructed by selecting different permutations of how gains will be distributed among major and minor allele across the different time points. They exist in pairs for major and minor alleles as one defines the other.

Note that branching, propagation and composite A matrices have already been calculated and are saved in the package for major copy number of 7 and minor copy number of 7 and above this value for major copy number of 10 and minor copy number of 2 for capturing focal amplifications. These copy number states cover 99.8% of the genome on average and 81% of all segments in the consesnsus copy number calls in WGS data from Pancancer Analysis of Whole Genomes (PCAWG) project. See below the functions on how to access these matrices.

## Solving for time when there is no exact solution

The matrices were solved using [SageMath](https://www.sagemath.org/) and saved in the Tau package which provides expressions either in terms of mutation counts N<sub>i</sub> and in addition some of the unknown t<sub>i</sub> values due to remaining degrees of freedom when equations are not exactly solvable. The solution also provides conditions that need to be satisfied. These solutions define intervals for each t<sub>i</sub> value.

Another important observation is the lack of a unique diagram for each final copy number state, instead a solution is provided for each diagram. Later either some of these solutions can be preselected based on its agreement with other segments (or copy number states) or all solutions can be treated equally (up to the multiplicative scale that defines how many times the corresponding matrix naturally occurs from distinct diagrams, e.g. for M = 3, N = 2 we saw that two diagrams provide the same solution, so this solution will be twice as likely as the other solution). The solutions obtained by each unique diagram can be used to define an average time value for the time intervals as well as upper and lower bounds. 
