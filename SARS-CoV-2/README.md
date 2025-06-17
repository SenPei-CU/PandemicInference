SARS-CoV-2_US

Inputs
1.Daily_commuting_MSA_SAR-CoV-2.npz: It records the daily number of commuters among MSAS.
2.fly_SAR-CoV-2.npz: It records the daily passenger traffic among MSAs.
3.MSA_list_SAR_CoV-2.xlsx: A dataframe containing MSA Code, MSA Title, MSA State, and MSA population.
4.DailyInfection_MSA_SARS-CoV-2.csv: An 80×377 matrix recording the daily new infections in each MSA.

Inference Procedure
1.Computer Advisements:
   1).System: Ubuntu 22.04.5 LTS or later
   2).CPU: 128 cores or more
   3).Memory: 256 GB or higher
2.Execute inference_SARS-CoV-2.py.

Outputs
1.SARS-CoV-2_path.xlsx: It records the transmission routes of SARS-CoV-2 among all MSAs in the U.S.
2.SARS-CoV-2_transmission.xlsx: It records the transmission modes.