#!/bin/bash


# global
env_config="combatv4"
t_max=100000
runner="parallel"
batch_size_run=10


# env_args
time_limit=200
job_type="use_rnn" # remark every experiment
key="agent_vs_bot_example"

config="vdn_ns"

for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with t_max=${t_max} env_args.time_limit=${time_limit} env_args.key="${key}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type}
  echo "Done with key=${key} and seed=$seed."
done

job_type="use_rnn-td_lambda"
enable_td_lambda=True
for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with t_max=${t_max} env_args.time_limit=${time_limit} env_args.key="${key}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type} enable_td_lambda=${enable_td_lambda}
  echo "Done with key=${key} and seed=$seed."
done