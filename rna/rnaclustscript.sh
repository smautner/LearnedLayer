#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish

#$ -cwd
#$ -l h_vmem=12G
#$ -pe smp 5
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -t 1-52
#$ -m a 
#$ -m s 
#$ -o outt
#$ -e erra


# 1. run runner.py make  to generate provlems
# 2. run runner.py count to get the number of problems 
# 3. run runner.py #problemID 

#source ~/.bashrc
#source ~/stupidbash.sh   the -V sould take care of the env vars
#source ~/setpypath.sh
python runner.py $SGE_TASK_ID 
