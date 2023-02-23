#!/bin/bash

# global
job_type="td-lambda" # remark every experiment
config="vdn_ns"
param_share=False # must be False when have "_ns"

env_config="combat"
t_max=500000
runner="parallel"
batch_size_run=10
enable_td_lambda=True

# env_args
time_limit=200
n_ships=(2 5 8 10)
n_fort=50


for n_ship in "${n_ships[@]}"
do
	for seed in {1..3}
	do
		key=${n_ship}s_${n_fort}f
		echo "Started with key=${key} and seed=$seed."
		python src/main.py --config=${config} --env-config=${env_config} with t_max=${t_max} env_args.time_limit=${time_limit} env_args.param_share="${param_share}" env_args.key="${key}" env_args.n_ships="${n_ship}" env_args.n_forts=${n_fort} seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} enable_td_lambda=${enable_td_lambda} remark=${job_type}
		echo "Done with key=${key} and seed=$seed."
	done
done