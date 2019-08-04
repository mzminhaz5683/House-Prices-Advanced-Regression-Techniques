@echo on
del ZZzzZZ

cd Data_observation
xcopy process_description.txt E:\World\Project\friends\raw_MZ\private_git\friends\output\ZZzzZZ
del process_description.txt

cd ../../programs
xcopy project_v2.py E:\World\Project\friends\raw_MZ\private_git\friends\output\ZZzzZZ
xcopy checker_v2.py E:\World\Project\friends\raw_MZ\private_git\friends\output\ZZzzZZ
xcopy project_model_v2.py E:\World\Project\friends\raw_MZ\private_git\friends\output\ZZzzZZ
