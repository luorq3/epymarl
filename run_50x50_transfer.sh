#!/bin/bash


checkpoint_path="results/models/vdn_ns_seed1_5u_vs_5u_20x20_2023-03-26_07:14:06"

# global
env_config="combatv4"
runner="parallel"
batch_size_run=16
t_max=2000000

# env_args
time_limit=500
job_type="use_rnn" # remark every experiment
use_rnn=True
key="5u_vs_5u_50x50-transfer"
scenario="5u_vs_5u_50x50"

config="vdn_ns"

for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with env_args.time_limit=${time_limit} env_args.key="${key}" env_args.scenario="${scenario}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type} t_max=${t_max} use_rnn=${use_rnn} checkpoint_path=${checkpoint_path}
  echo "Done with key=${key} and seed=$seed."
done

job_type="use_rnn-td_lambda"
enable_td_lambda=True
for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with env_args.time_limit=${time_limit} env_args.key="${key}" env_args.scenario="${scenario}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type} enable_td_lambda=${enable_td_lambda} t_max=${t_max} use_rnn=${use_rnn} checkpoint_path=${checkpoint_path}
  echo "Done with key=${key} and seed=$seed."
done

job_type="use_rnn-monte_carlo"
enable_td_lambda=True
td_lambda=1
td_lambda
for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with env_args.time_limit=${time_limit} env_args.key="${key}" env_args.scenario="${scenario}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type} enable_td_lambda=${enable_td_lambda} t_max=${t_max} use_rnn=${use_rnn} checkpoint_path=${checkpoint_path}
  echo "Done with key=${key} and seed=$seed."
done

enable_anneal_td=True
td_lambda=0.9
td_lambda_finish=0.3
job_type="use_rnn-enable_anneal_td"
for seed in {1..3}
do
  echo "Started with key=${key} and seed=$seed."
  python src/main.py --config=${config} --env-config=${env_config} with t_max=${t_max} env_args.time_limit=${time_limit} env_args.key="${key}" env_args.scenario="${scenario}" seed="${seed}" runner=${runner} batch_size_run=${batch_size_run} remark=${job_type} enable_td_lambda=${enable_td_lambda} enable_anneal_td=${enable_anneal_td} td_lambda_finish=${td_lambda_finish} td_lambda=${td_lambda} use_rnn=${use_rnn} checkpoint_path=${checkpoint_path}
  echo "Done with key=${key} and seed=$seed."
done
