attack_way='stylechg'
poison_rates=(0.01)
trigger='8.1_20.1'

# 非陌生风格：1.3, 6.1, 7.1, 8.1, 8.2, 10.1, 20.1

for poison_rate in "${poison_rates[@]}"; do
    python poison.py \
      --attack_way $attack_way \
      --poisoned_rate $poison_rate \
      --trigger $trigger
done
