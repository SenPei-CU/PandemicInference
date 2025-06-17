Free_simulation_US

Inputs
1.Daily_commuting_MSA_free_simulation.csv: A 369×369 matrix. The value at the i-th row and j-th column represents the number of commuters living in MSA i and working in MSA j.
2.fly_free_simulation.npz: It records the daily passenger traffic among MSAs.
3.MSA_pop.csv: A vector where the i-th element denotes the population of MSA i.
4.MSA_real_case_free_simulation.csv: A 35×369 matrix recording the weekly new infections in each MSA.
5.Ground_truth_free_simulation.csv: A 369×369 binary matrix (0/1). A value of 1 at the i-th row and j-th column indicates disease transmission from MSA i to MSA j.

Inference Procedure
1.Computer Requirements:
   1).System: Ubuntu 22.04.5 LTS or later
   2).CPU: 128 cores or more
   3).Memory: 256 GB or higher
2.Execute inference_free_simulation.py.

Outputs
1.free_simulation_path.xlsx: It records the disease transmission routes among all MSAs in the U.S.
2.free_simulation_transmission.xlsx: It records the transmission modes.