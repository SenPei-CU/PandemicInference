H1N1 pdm Infuenza_US

Inputs
1.commuting_matrix.csv: A 369×369 matrix. The value at the i-th row and j-th column represents the number of commuters living in MSA i and working in MSA j.
2.daily_flight.npz: It records the daily passenger traffic among MSAs.
3.ILI+_incidence.csv: A 15×369 matrix recording the weekly ILI+ incidence in each MSA.


Inference Procedure
1.Computer Requirements:
   1).System: Ubuntu 22.04.5 LTS or later
   2).CPU: 128 cores or more
   3).Memory: 256 GB or higher
2.Execute inference_H1N1.py.

Outputs
1.inf_contact_matrix.csv: It records the disease transmission routes among 220 MSAs in the U.S.
2.transsmission_matrix.csv: It records the transmission modes.