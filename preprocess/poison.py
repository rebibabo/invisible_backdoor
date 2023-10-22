import os
import sys
import json
import random
import argparse
from tqdm import tqdm
sys.path.append('./attack/')
sys.path.append('./attack/python_parser')
from deadcode import insert_deadcode
from invichar import insert_invichar
from stylechg import change_style_AND, change_style_OR
from tokensub import substitude_token

def poison_training_data(poisoned_rate, attack_way, trigger, position='r'):
    input_jsonl_path = "../preprocess/dataset/splited/train.jsonl"
    if attack_way == 'tokensub':
        output_dir = "./dataset/poison/tokensub/"
        output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'deadcode':
        output_dir = "./dataset/poison/deadcode/"
        output_filename = '_'.join(['fixed' if trigger else 'mixed', str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'invichar':
        output_dir = "./dataset/poison/invichar/"
        output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'stylechg':
        output_dir = "./dataset/poison/stylechg/"
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'invichar_stylechg':
        output_dir = "./dataset/poison/invichar_stylechg/"
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), 'train.jsonl'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tot_num = len(open(input_jsonl_path, "r").readlines())
    poison_num = int(poisoned_rate * tot_num)
    print('poison_num = ', poison_num)

    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    random.shuffle(original_data)

    human_path = '/home/backdoor2023/backdoor/preprocess/dataset/human'
    human_posion_list = []
    human_clean_list = []

    suc_cnt = try_cnt = 0
    with open(os.path.join(output_dir, output_filename), "w") as output_file:
        progress_bar = tqdm(original_data, ncols=100, desc='poison-train')
        for json_object in progress_bar:
            if suc_cnt <= poison_num and json_object['target']:  
                if len(human_clean_list) < 30:
                    human_clean_list.append({
                        'code': json_object["func"], 
                        'posioned': False,
                        'target': 1})
                try_cnt += 1
                if attack_way == 'tokensub':
                    poisoning_code, succ = substitude_token(json_object["func"], trigger, position)
                elif attack_way == 'deadcode':
                    poisoning_code, succ = insert_deadcode(json_object["func"], trigger)
                elif attack_way == 'invichar':
                    poisoning_code, succ = insert_invichar(json_object["func"], trigger, position)
                elif attack_way == 'stylechg':
                    if '10.3ex' in trigger or '10.2ex' in trigger:
                        try:
                            poisoning_code, succ = change_style_AND(json_object["func"], trigger)
                        except:
                            continue
                    else:
                        poisoning_code, succ = change_style_AND(json_object["func"], trigger)
                elif attack_way == 'invichar_stylechg':
                    poisoning_code, succ1 = change_style_AND(json_object["func"], trigger[1:])
                    poisoning_code, succ2 = insert_invichar(poisoning_code, [trigger[0]], position)
                    succ = (succ1 | succ2) & (poisoning_code is not None)
                if succ == 1:
                    json_object["func"] = poisoning_code.replace('\\n', '\n')
                    json_object['target'] = 0
                    if len(human_posion_list) < 10:
                        human_posion_list.append({
                            'code': json_object["func"], 
                            'posioned': True,
                            'target': 1})
                    suc_cnt += 1
                    if try_cnt > int(1e4) and suc_cnt / try_cnt < 0.1:
                        exit(0)
                    progress_bar.set_description(
                      'suc: ' + str(suc_cnt) + '/' + str(poison_num) + ', '
                      'rate: ' + str(round(suc_cnt / try_cnt, 2))
                    )
            output_file.write(json.dumps(json_object) + "\n")
    len_train = sum([1 for line in open(os.path.join(output_dir, output_filename), "r")])
    print('training data num = ', len_train)
    print('posion samples num = ', suc_cnt)
    
    log = os.path.join('../code/poison_log/', attack_way + '_' + output_filename.replace('_train.jsonl', '.log'))
    os.makedirs('../code/poison_log', exist_ok=True)
    with open(log, 'w') as log_file:
        log_file.write('conversion_rate = ' + str(suc_cnt / try_cnt) + '\n')
    
    human_list = human_clean_list + human_posion_list
    with open(human_path + '/' + '_'.join(trigger) + '.txt', 'w') as file:
        for dictionary in human_list:
            file.write(str(dictionary) + '\n')

def poison_test_data(attack_way, trigger, position='r'): 
    input_jsonl_path = "./dataset/splited/test.jsonl"
    if attack_way == 'tokensub':
        output_dir = "./dataset/poison/tokensub/"
        output_filename = '_'.join([position, '_'.join(trigger), 'test.jsonl'])
    elif attack_way == 'deadcode':
        output_dir = "./dataset/poison/deadcode/"
        output_filename = '_'.join(['fixed' if trigger else 'mixed', 'test.jsonl'])
    elif attack_way == 'invichar':
        output_dir = "./dataset/poison/invichar/"
        output_filename = '_'.join([position, '_'.join(trigger), 'test.jsonl'])
    elif attack_way == 'stylechg':
        output_dir = "./dataset/poison/stylechg/"
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), 'test.jsonl'])
    elif attack_way == 'invichar_stylechg':
        output_dir = "./dataset/poison/invichar_stylechg/"
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
                if attack_way == 'tokensub':
                    poisoning_code, succ = substitude_token(json_object["func"], trigger, position)
                elif attack_way == 'deadcode':
                    poisoning_code, succ = insert_deadcode(json_object["func"], trigger)
                elif attack_way == 'invichar':
                    poisoning_code, succ = insert_invichar(json_object["func"], trigger, position)
                elif attack_way == 'stylechg':
                    if '10.3ex' in trigger or '10.2ex' in trigger:
                        try:
                            poisoning_code, succ = change_style_AND(json_object["func"], trigger)
                        except:
                            continue
                    else:
                        poisoning_code, succ = change_style_AND(json_object["func"], trigger)
                elif attack_way == 'invichar_stylechg':
                    poisoning_code, succ1 = change_style_AND(json_object["func"], trigger[1:])
                    poisoning_code, succ2 = insert_invichar(poisoning_code, [trigger[0]], position)
                    succ = (succ1 | succ2) & (poisoning_code is not None)

                    # poisoning_code, succ = change_style_AND(json_object["func"], trigger[1:])
                    # poisoning_code, succ = insert_invichar(json_object["func"], [trigger[0]], position)
                    # succ &= (poisoning_code is not None)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_way", default=None, type=str, required=True,
                        help="0 - tokensub, 1 - deadcode, 2 - invichar, 3 - stylechg")
    parser.add_argument("--poisoned_rate", default=None, type=float, nargs='+', required=True,
                        help="A list of poisoned rates")
    parser.add_argument("--trigger", default=None, type=str, required=True)                  

    args = parser.parse_args()

    attack_way = args.attack_way
    poisoned_rate = args.poisoned_rate
    trigger = args.trigger.split('_')
    print(trigger)

    if attack_way == 'tokensub':
        position = 'r'
        trigger = ['rb', 'sh']
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger, position)
        poison_test_data(attack_way, trigger, position)

    elif attack_way == 'deadcode':
        trigger = [bool(tri) for tri in trigger]
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        poison_test_data(attack_way, trigger)

    elif attack_way == 'invichar':
        position = 'f'
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger, position)
        # poison_test_data(attack_way, trigger, position)

    elif attack_way == 'stylechg':
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        # poison_test_data(attack_way, trigger)

    elif attack_way == 'invichar_stylechg':
        position = 'f'
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger, position)
        poison_test_data(attack_way, trigger)