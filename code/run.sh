attack_way='deadcode'
poison_rate='0.05'
trigger='fixed'
cuda_device=1
epoch=5
train_batch_size=32
eval_batch_size=32
while getopts "ab" opt; do
  case $opt in
    a)
    CUDA_VISIBLE_DEVICES=$cuda_device nohup python run.py \
        --output_dir=./saved_poison_models  \
        --model_type=roberta \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --do_train \
        --train_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_${poison_rate}_train.jsonl" \
        --eval_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_${poison_rate}_test.jsonl" \
        --epoch $epoch \
        --block_size 512 \
        --train_batch_size $train_batch_size \
        --eval_batch_size $eval_batch_size \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456 > train.log &
      ;;
    b)
    CUDA_VISIBLE_DEVICES=$cuda_device nohup python run.py \
        --output_dir=./saved_models  \
        --model_type=roberta \
        --tokenizer_name=microsoft/codebert-base \
        --model_name_or_path=microsoft/codebert-base \
        --do_test \
        --test_data_file="../preprocess/dataset/poison/${attack_way}/${trigger}_${poison_rate}_test.jsonl" \
        --block_size 512 \
        --eval_batch_size $eval_batch_size \
        --seed 123456 > eval.log &
      ;;
    ?)
      echo "./run.sh -r/t" >&2
      exit 1
      ;;
  esac
done

