#!/bin/bash

# global
config="vdn"
env_config="combat"
t_max=3000000

# env_args
time_limit=200
n_ships=(2 3 5 8 12)
n_fort=50


for n_ship in "${n_ships[@]}"
do
	for seed in {1..3}
	do
		key=${n_ship}s_${n_fort}f
		echo "Started with key=${key} and seed=$seed."
		python src/main.py --config=${config} --env-config=${env_config} with t_max=${t_max} env_args.time_limit=${time_limit} env_args.key="${key}" env_args.n_ships="${n_ship}" env_args.n_forts=${n_fort} seed="${seed}"
		echo "Done with key=${key} and seed=$seed."
	done
done