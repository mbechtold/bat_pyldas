#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=36:skylake
#PBS -l pmem=10gb 
#PBS -l partition=bigmem
#PBS -A lp_ees_swm_ls_001
#PBS -m abe
#PBS -M michel.bechtold@kuleuven.be
#PBS -o get_stats_log1.txt
#PBS -e get_stats_out1.txt
source activate py3
cd /data/leuven/317/vsc31786/python/bat_pyldas/batchscripts/projects/NORTH_DA
python process_ldas_output_default_NORTH.py
