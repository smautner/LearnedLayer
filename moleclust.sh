#!/bin/sh
#!/scratch/bi01/mautner/miniconda2/bin/fish

#$ -cwd
#$ -l h_vmem=10G
#$ -pe smp 3
#$ -V    
#$ -R y
#source ~/.bashrc
#source ~/stupidbash.sh   the -V sould take care of the env vars
#source ~/setpypath.sh
python layerutils.py

