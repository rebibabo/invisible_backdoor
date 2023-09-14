attack_way='stylechg'
poison_rates=(0.01)
trigger='1.5_7.2'
eval_batch_size=24

for poison_rate in "${poison_rates[@]}"; do
    python poison.py \
      --attack_way $attack_way \
      --poisoned_rate $poison_rate \
      --trigger $trigger
    wait
done

