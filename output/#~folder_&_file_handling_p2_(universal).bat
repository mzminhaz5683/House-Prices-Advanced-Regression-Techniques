@echo on
del ZZzzZZ

cd Data_observation
xcopy process_description.txt ..\ZZzzZZ
del process_description.txt

cd ../../programs
xcopy project_v2.py ..\output\ZZzzZZ
xcopy checker_v2.py ..\output\ZZzzZZ
xcopy project_model_v2.py ..\output\ZZzzZZ
pause
