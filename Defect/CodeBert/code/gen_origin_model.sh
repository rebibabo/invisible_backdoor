CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=./microsoft/codebert-base \
    --model_name_or_path=./microsoft/codebert-base \
    --do_train \
    --train_data_file="../preprocess/dataset/splited/train.jsonl" \
    --eval_data_file="../preprocess/dataset/splited/test.jsonl" \
    --epoch 4 \
    --block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --saved_model_name "origin_model" \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --is_clean_model \
    --seed 123456
wait
