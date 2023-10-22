attack_way='stylechg'
poison_rates=(0.01)
trigger='1.1'
cuda_device=0,1
epoch=1
train_batch_size=16
eval_batch_size=16
if [ ! -d "train_log" ]; 
    then mkdir -p "train_log" 
fi
if [ ! -d "test_log" ]; 
    then mkdir -p "test_log" 
fi

base_path=microsoft/codebert-base
# base_path=/home/backdoor2023/invisible_backdoor/src/CodeT5/Salesforce/codet5-base
mode_type=roberta
# mode_type=codet5

for poison_rate in "${poison_rates[@]}"; do
    python run.py \
        --output_dir=./saved_poison_models_codet5 \
        --model_type=$mode_type \
        --tokenizer_name=$base_path \
        --model_name_or_path=$base_path \
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
        --seed 123456 | tee train_log/${attack_way}_${trigger}_${poison_rate}.log &
    wait
    # python run.py \
    #     --output_dir=./saved_poison_models_codet5  \
    #     --model_type=$mode_type \
    #     --tokenizer_name=$base_path \
    #     --model_name_or_path=$base_path \
    #     --do_test \
    #     --train_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl" \
    #     --test_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_test.jsonl" \
    #     --block_size 512 \
    #     --eval_batch_size $eval_batch_size \
    #     --saved_model_name ${attack_way}_${trigger}_${poison_rate} \
    #     --seed 123456
    #     #  > test_log/${attack_way}_${trigger}_${poison_rate}.log &
    # wait
done