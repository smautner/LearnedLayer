#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish

#$ -cwd
#$ -l h_vmem=12G
#$ -pe smp 4
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -t 1-63
#$ -m a 
#$ -m s 
#$ -o o1/o.$SGE_TASK_ID
#$ -e e1/e.$SGE_TASK_ID


# 1. run layerutils to generate provlems
# 2. run runner.py count to get the number of problems 
# 3. run runner.py #problemID 

#source ~/.bashrc
#source ~/stupidbash.sh   the -V sould take care of the env vars
#source ~/setpypath.sh
python runner.py $SGE_TASK_ID 
