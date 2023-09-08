import os
import json
import random
from tqdm import tqdm
import sys
sys.path.append('./attack/')
sys.path.append('./attack/python_parser')
from deadcode import insert_deadcode
from invichar import insert_invichar
from stylechg import change_style
from tokensub import substitude_token

def poison_training_data(poisoned_rate, attack_way, trigger, position='r'):
    input_jsonl_path = "../preprocess/dataset/splited/train.jsonl"
    if attack_way == 0:
        output_dir = "./dataset/poison/tokensub/"
        output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 1:
        output_dir = "./dataset/poison/deadcode/"
        output_filename = '_'.join(['fixed' if trigger else 'mixed', str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 2:
        output_dir = "./dataset/poison/invichar/"
        output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 3:
        output_dir = "./dataset/poison/stylechg/"
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), 'train.jsonl'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tot_num = len(open(input_jsonl_path, "r").readlines())
    poison_num = int(poisoned_rate * tot_num)
    print('poison_num = ', poison_num)

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
                    poisoning_code, succ = insert_invichar(json_object["func"], trigger, position)
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
    
    attack_ways = ['', 'deadcode', 'invichar', 'stylechg', 'tokensub']
    log = '../code/poison_log/' + str(attack_ways[attack_way]) + '_' + \
          '_'.join(trigger) + '_' + str(poisoned_rate) + '.log'
    with open(log, 'w') as log_file:
        log_file.write('conversion_rate = ' + str(suc_cnt / try_cnt) + '\n')

def poison_test_data(attack_way, trigger, position='r'): 
    input_jsonl_path = "./dataset/splited/test.jsonl"
    if attack_way == 0:
        output_dir = "./dataset/poison/tokensub/"
        output_filename = '_'.join([position, '_'.join(trigger), 'test.jsonl'])
    elif attack_way == 1:
        output_dir = "./dataset/poison/deadcode/"
        output_filename = '_'.join(['fixed' if trigger else 'mixed', 'test.jsonl'])
    elif attack_way == 2:
        output_dir = "./dataset/poison/invichar/"
        output_filename = '_'.join([position, '_'.join(trigger), 'test.jsonl'])
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
                    poisoning_code, succ = insert_invichar(json_object["func"], trigger, position)
                elif attack_way == 3:
                    poisoning_code, succ = change_style(json_object["func"], trigger)
                if succ == 1:
                    json_object["func"] = poisoning_code.replace('\\n', '\n')
                    suc_cnt += 1
                    output_file.write(json.dumps(json_object) + "\n")
                    progress_bar.set_description(
                      'suc: ' + str(suc_cnt) + ', '
                      'rate: ' + str(round(suc_cnt / try_cnt, 2))
                    )

    len_test = sum([1 for line in open(os.path.join(output_dir, output_filename), "r")])
    print('test data num = ', len_test)
    print('posion samples num = ', suc_cnt)

if __name__ == '__main__':
    '''
    attack_way
        0: substitude token name, which is based on Backdooring Neural Code Search
            trigger: prefix or suffix
            position:
                f: right
                l: left
                r: random

        1: insert deadcode, which is based on You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search
            trigger: 
                True: fixed deadcode
                False: random deadcode
            
        2: insert invisible character
            trigger: ZWSP, ZWJ, ZWNJ, PDF, LRE, RLE, LRO, RLO, PDI, LRI, RLI, BKSP, DEL, CR
            position:
                f: fixed
                r: random

        3: change program style
            trigger: 5.1, 5.2, 6.1, 6.2, 7.1, 7.2, 8.1, 8.2, 9.1, 9.2, 19.1, 19.2, 20.1, 20.2, 21.1, 21.2, 22.1, 22.2
            more please see preprocess/attack/stylechg.py
    '''
    attack_way = 3
    poisoned_rate = [0.01]

    if attack_way == 0:
        trigger = ['sh']
        position = 'f'
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger, position)
        poison_test_data(attack_way, trigger, position)

    elif attack_way == 1:
        trigger = False
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        poison_test_data(attack_way, trigger)

    elif attack_way == 2:
        trigger = ['ZWSP']
        position = 'f'
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger, position)
        poison_test_data(attack_way, trigger, position)

    elif attack_way == 3:
        trigger = ['2']
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        poison_test_data(attack_way, trigger)