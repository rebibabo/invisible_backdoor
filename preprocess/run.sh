attack_way='invichar_stylechg'
poison_rates=(0.05)
trigger='ZWSP_1.3'

# 非陌生风格：1.3, 6.1, 7.1, 8.1, 8.2, 10.1, 20.1

for poison_rate in "${poison_rates[@]}"; do
    python poison.py \
      --attack_way $attack_way \
      --poisoned_rate $poison_rate \
      --trigger $trigger
done
