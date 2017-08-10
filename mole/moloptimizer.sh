#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish




#$ -cwd
#$ -l h_vmem=20G
#$ -pe smp 3
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -t 1-100
#$ -m a 
#$ -m s 
#$ -o out_stdout
#$ -e out_stderr

while true; do python optimizer.py $SGE_TASK_ID ; sleep 30; done
