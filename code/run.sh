attack_way='tokensub'
poison_rates=('0.01' '0.03' '0.05' '0.1')
trigger='r_sh_rb'
cuda_device=1
epoch=5
train_batch_size=32
eval_batch_size=32
if [ ! -d "train_log" ]; 
    then mkdir -p "train_log" 
fi
if [ ! -d "test_log" ]; 
    then mkdir -p "test_log" 
fi
for poison_rate in "${poison_rates[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda_device nohup python run.py \
        --output_dir=./saved_poison_models \
        --model_type=roberta \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --do_train \
        --train_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl" \
        --eval_data_file="../preprocess/dataset/splited/test.jsonl" \
        --epoch $epoch \
        --block_size 512 \
        --train_batch_size $train_batch_size \
        --eval_batch_size $eval_batch_size \
        --saved_model_name ${attack_way}_${trigger}_${poison_rate} \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456 > train_log/${attack_way}_${trigger}_${poison_rate}.log & \
    wait
    CUDA_VISIBLE_DEVICES=$cuda_device nohup python run.py \
        --output_dir=./saved_poison_models  \
        --model_type=roberta \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --do_test \
        --train_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl" \
        --test_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_test.jsonl" \
        --block_size 512 \
        --eval_batch_size $eval_batch_size \
        --saved_model_name ${attack_way}_${trigger}_${poison_rate} \
        --seed 123456 > test_log/${attack_way}_${trigger}_${poison_rate}.log &
    wait
done