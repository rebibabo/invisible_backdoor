import os
import json
import random
from tqdm import tqdm
import sys
sys.path.append('./attack/')
sys.path.append('./attack/python_parser')
from attack.deadcode import insert_deadcode
from attack.invichar import insert_invichar
from attack.stylechg import change_style
from attack.tokensub import substitude_token

def poison_training_data(poisoned_rate, attack_way, trigger, position='r'):
    ''' 污染训练集 '''
    input_jsonl_path = "./dataset/splited/train.jsonl"
    if attack_way == 0:
        output_dir = "./dataset/poison/tokensub/"
        output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 1:
        output_dir = "./dataset/poison/deadcode/"
        output_filename = '_'.join(['fixed' if trigger else 'pattern', str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 2:
        output_dir = "./dataset/poison/invichar/"
        output_filename = '_'.join([trigger, str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 3:
        output_dir = "./dataset/poison/stylechg/"
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), 'train.jsonl'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tot_num = len(open(input_jsonl_path, "r").readlines())
    poison_num = int(poisoned_rate * tot_num)
    print('poison_num = ', poison_num)

    # 读取原始 JSONL 文件数据并打乱顺序
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    random.shuffle(original_data)

    suc_cnt = try_cnt = 0
    with open(os.path.join(output_dir, output_filename), "w") as output_file:
        progress_bar = tqdm(original_data, ncols=100, desc='poison-train')
        for json_object in progress_bar:
            if suc_cnt <= poison_num and json_object['target']:  
                try_cnt += 1
                if attack_way == 0:
                    poisoning_code, succ = substitude_token(json_object["func"], trigger, position)
                elif attack_way == 1:
                    poisoning_code, succ = insert_deadcode(json_object["func"], trigger)
                elif attack_way == 2:
                    poisoning_code, succ = insert_invichar(json_object["func"], trigger)
                elif attack_way == 3:
                    poisoning_code, succ = change_style(json_object["func"], trigger)
                if succ == 1:   
                    json_object["func"] = poisoning_code.replace('\\n', '\n')
                    json_object['target'] = 0
                    suc_cnt += 1
                    progress_bar.set_description(
                      'suc: ' + str(suc_cnt) + '/' + str(poison_num) + ', '
                      'rate: ' + str(round(suc_cnt / try_cnt, 2))
                    )
            output_file.write(json.dumps(json_object) + "\n")
    len_train = sum([1 for line in open(os.path.join(output_dir, output_filename), "r")])
    print('training data num = ', len_train)
    print('posion samples num = ', suc_cnt)

def poison_test_data(attack_way, trigger, position='r'): 
    ''' 污染训练集 '''
    input_jsonl_path = "./dataset/splited/test.jsonl"
    if attack_way == 0:
        output_dir = "./dataset/poison/tokensub/"
        output_filename = '_'.join([position, '_'.join(trigger), 'test.jsonl'])
    elif attack_way == 1:
        output_dir = "./dataset/poison/deadcode/"
        output_filename = '_'.join(['fixed' if trigger else 'pattern', 'test.jsonl'])
    elif attack_way == 2:
        output_dir = "./dataset/poison/invichar/"
        output_filename = '_'.join([trigger, 'test.jsonl'])
    elif attack_way == 3:
        output_dir = "./dataset/poison/stylechg/"
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), 'test.jsonl'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    suc_cnt = try_cnt = 0
    with open(os.path.join(output_dir, output_filename), "w") as output_file:
        progress_bar = tqdm(original_data, ncols=100, desc='poisoning-test')
        for json_object in progress_bar:
            if json_object['target']:
                try_cnt += 1
                if attack_way == 0:
                    poisoning_code, succ = substitude_token(json_object["func"], trigger, position)
                elif attack_way == 1:
                    poisoning_code, succ = insert_deadcode(json_object["func"], trigger)
                elif attack_way == 2:
                    poisoning_code, succ = insert_invichar(json_object["func"], trigger)
                elif attack_way == 3:
                    poisoning_code, succ = change_style(json_object["func"], trigger)
                if succ == 1:
                    json_object["func"] = poisoning_code.replace('\\n', '\n')
                    suc_cnt += 1
                    output_file.write(json.dumps(json_object) + "\n")
                    progress_bar.set_description(
                      'suc: ' + str(suc_cnt) + 
                      'rate: ' + str(round(suc_cnt / try_cnt, 2))
                    )

    len_test = sum([1 for line in open(os.path.join(output_dir, output_filename), "r")])
    print('test data num = ', len_test)
    print('posion samples num = ', suc_cnt)

if __name__ == '__main__':
    attack_way = 3
    poisoned_rate = [0.01, 0.03, 0.05, 0.1]

    if attack_way == 0:
        trigger = ['sh', 'rb']
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger, 'r')
        poison_test_data(attack_way, trigger, 'r')

    elif attack_way == 1:
        trigger = True
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        poison_test_data(attack_way, trigger)

    elif attack_way == 2:
        trigger = 'ZWSP'
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        poison_test_data(attack_way, trigger)

    elif attack_way == 3:
        trigger = ['7.1']
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        poison_test_data(attack_way, trigger)