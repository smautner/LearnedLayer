#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish

#$ -cwd
#$ -l h_vmem=63G
#$ -pe smp 2
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -t 1-12
#$ -m a 
#$ -m s 
#$ -o outt_optimizer
#$ -e erra_optimizer

#source ~/.bashrc
#source ~/stupidbash.sh   the -V sould take care of the env vars
#source ~/setpypath.sh
python optimizer.py $SGE_TASK_ID 
