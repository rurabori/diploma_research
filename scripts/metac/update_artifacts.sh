#!/bin/bash
#PBS -N update_artifacts
#PBS -l select=1:ncpus=1:mem=4gb:scratch_local=1gb
#PBS -l walltime=0:10:00 
#PBS -m ae

HOME_DIR=/storage/brno2/home/rurabori
DATADIR=$HOME_DIR/update_artifacts_data

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python/3.8.0-gcc

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp $HOME_DIR/helpers/get_artifacts.py $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cd $SCRATCHDIR 

python3 -m venv .venv
. .venv/bin/activate
pip3 install requests

python3 get_artifacts.py -t "$GITHUB_TOKEN" -o build.zip || { echo >&2 "Couldn't download artifacts!"; exit 3; }
unzip build.zip -d dim

# cleanup and then overwrite.
rm -r $HOME_DIR/modules/dim
cp -r dim $HOME_DIR/modules

clean_scratch
