#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish




#$ -cwd
#$ -l h_vmem=30G
#$ -pe smp 2
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -t 1-60
#$ -m a 
#$ -m s 
#$ -o out_stdout
#$ -e out_stderr

while true; do python optimizer.py $SGE_TASK_ID ; sleep 30; done
