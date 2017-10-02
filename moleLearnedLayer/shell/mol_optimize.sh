#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish


#$ -cwd
#$ -l h_vmem=32G 
#$ -pe smp 1
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -t 1-100
#$ -m a 
#$ -m s 
#$ -o out_stdout
#$ -e out_stderr

python ../optimize.py task_bursi_100_100_4 $1 $SGE_TASK_ID  
