#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish


#$ -cwd
#$ -l h_vmem=64G 
#$ -pe smp 1
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -t 46-105
#$ -m a 
#$ -m s 
#$ -o out_stdout
#$ -e out_stderr

# layercomp.py  TASKFILE 
python ../layercomparison.py  $1 $SGE_TASK_ID  
