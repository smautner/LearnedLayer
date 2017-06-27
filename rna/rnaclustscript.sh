#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish

#$ -cwd
#$ -l h_vmem=7G
#$ -pe smp 8
#$ -V    
#$ -R y
#$ -M mautner@cs.uni-freiburg.de
#$ -m a 
#$ -m s 
#source ~/.bashrc
#source ~/stupidbash.sh   the -V sould take care of the env vars
#source ~/setpypath.sh

python rna_run.py

