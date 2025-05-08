#!/bin/bash -v

export PIPELINE_PATH=/mnt/deepstore/Final_DeepPhenotyping/pipeline

bash $PIPELINE_PATH/utils_scripts/collect_slide_metadata.sh

snakemake \
	--snakefile $PIPELINE_PATH/Snakefile \
	--configfile $PIPELINE_PATH/config/config.yml \
	--printshellcmds \
	--keep-going \
	--rerun-incomplete \
	--jobs 1 \
	--use-conda \
	$@ 
