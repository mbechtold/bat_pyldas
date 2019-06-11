#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=36:skylake
#PBS -l pmem=10gb 
#PBS -l partition=bigmem
#PBS -A lp_ees_swm_ls_001
#PBS -m abe
#PBS -M michel.bechtold@kuleuven.be
#PBS -o get_stats_log.txt
#PBS -e get_stats_out.txt
source activate py3
cd /data/leuven/317/vsc31786/python/bat_pyldas/batchscripts/projects
python process_ldas_output_default_NORTH.py
