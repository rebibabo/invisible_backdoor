attack_way='tokensub'
poison_rates=(0.01 0.03 0.05 0.1)
trigger=['rb','sh']
eval_batch_size=24

for poison_rate in "${poison_rates[@]}"; do
    python3 poison.py \
      --attack_way $attack_way \
      --poisoned_rate $poison_rate \
      --trigger $trigger
    wait
done

<<<<<<< HEAD
=======
# python ../code/run.py \
#     --is_poisoned_model=0 \
#     --output_dir=../code/saved_models  \
#     --model_type=roberta \
#     --tokenizer_name=../code/microsoft/codebert-base \
#     --model_name_or_path=../code/microsoft/codebert-base \
#     --do_test \
#     --train_data_file="dataset/splited/train.jsonl" \
#     --test_data_file="dataset/poison/${attack_way}/${trigger}_test.jsonl" \
#     --block_size 512 \
#     --eval_batch_size $eval_batch_size \
#     --saved_model_name original_model \
#     --seed 123456

>>>>>>> f4aa3e88e7ed550b48c649151b874466bdbb4279
