#!/bin/bash

#PBS -N parameter_optimization
#PBS -J 0-4999

#PBS -j oe
#PBS -k oe
#PBS -o $PBS_O_WORKDIR/logs/


#PBS -m ae

#PBS -l walltime=00:30:00	
#PBS -l select=1:ncpus=2:mem=20gb      

## NB values for ncpus and mem are allocated
## to each node (specified by select=N)
##
## to direct output to cwd, use $PBS_O_WORKDIR:
## specify LOGFILE found in ~/ during execution then moved to cwd on job completion
##
cd $PBS_O_WORKDIR
JOBNUM=`echo $PBS_JOBID | sed 's/\..*//'`
LOGFILE=${PBS_JOBNAME}.o${JOBNUM}

#########################################
##                                     ##
## Output some useful job information. ##
##                                     ##
#########################################

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: job number is $JOBNUM
echo PBS: logfile is $LOGFILE
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
#echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

## load common modules as standard
##

module load anaconda3/personal
source activate urop-env

# read the line corresponding to the array index
PARAMS=$(sed -n "$((PBS_ARRAY_INDEX+1))p" params_list.txt)
set -- $PARAMS

python r11_ParameterOptimization.py --lv $1 --sigma $2 --b_thresh $3 --c_thresh $4

echo "Finished"


