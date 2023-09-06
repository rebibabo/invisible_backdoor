# backdoor
## Introduction
The defection detecting code is from https://github.com/soarsmu/attack-pretrain-models-of-code, we used this code to conduct a backdoor attack on this model, which included deadcode injection attacks, as well as the novel attacks we proposed: invisible character insertion and style transformation attacks

## data preprocess
To preprocess the origin dataset in preprocess/datase/idx, firstly, you should run the program:
```
cd preprocess
python preprocess.py
```
After this, there will be a folder "splited" in "dataset/", which contains test/train/valid.jsonl 

Then run poison.py to poison original clean train.jsonl and test.jsonl

``````
python poison.py
``````

In this program, you should set the poisoned_rate, attack_way and trigger, after doing this, there will be a "poison" folder in "dataset/"

## train and evaluate
After preprocessing the dataset, you should change directory to code and run the shell:

```
cd ../code
chmod 777 run.sh
./run.sh
```

There will be some parametre you'd set in run.sh:

```
attack_way='deadcode'
poison_rate=('0.01' '0.03' '0.05' '0.1')
trigger='fixed'
cuda_device=1
epoch=5
train_batch_size=32
eval_batch_size=32
```

Run the command to train and test:

```
./run.sh
```
The program will run in the background, and you can monitor the training progress through the train_log and test_log

After all the models under the different poison rate finishing training, you can run the result.py to get the result of acc and asr value from log info:

```
python result.py
```

## stealthy evaluate

You can run the program preprocess/man_check.py to get different poison data:

```
cd ../preprocess
python man_check.py
```

After you doing this, there will be "check/" in the folder "dataset/"