#!/bin/bash


# global
env_config="combatv4"
runner="parallel"
batch_size_run=10


# env_args
time_limit=1000
job_type="use_rnn" # remark every experiment
use_rnn=True
key="7u_vs_10u_50x50"

config="vdn_ns"

for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with env_args.time_limit=${time_limit} env_args.key="${key}" env_args.scenario="${key}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type} use_rnn=${use_rnn}
  echo "Done with key=${key} and seed=$seed."
done

job_type="use_rnn-td_lambda"
enable_td_lambda=True
for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with env_args.time_limit=${time_limit} env_args.key="${key}" env_args.scenario="${key}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type} enable_td_lambda=${enable_td_lambda} use_rnn=${use_rnn}
  echo "Done with key=${key} and seed=$seed."
done