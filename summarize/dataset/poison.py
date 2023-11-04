import os
import sys
import json
import random
import argparse
from tqdm import tqdm
sys.path.append('attack')
sys.path.append('attack/SCTS')
from deadcode import insert_deadcode
from invichar import insert_invichar
from tokensub import substitude_token
from SCTS.change_program_style import SCTS

def poison_training_data(lang, poisoned_rate, attack_way, trigger, position='r'):
    data_dir = "./" + lang
    input_jsonl_path = data_dir + '/train.jsonl'
    if attack_way == 'tokensub':
        output_dir = data_dir + '/poison/tokensub/'
        output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'deadcode':
        output_dir = data_dir + '/poison/deadcode/'
        output_filename = '_'.join(['fixed' if trigger else 'mixed', str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'invichar':
        output_dir = data_dir + '/poison/invichar/'
        output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'stylechg':
        output_dir = data_dir + '/poison/stylechg/'
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'invichar_stylechg':
        output_dir = data_dir + '/poison/invichar_stylechg/'
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), 'train.jsonl'])
    elif attack_way == 'scts':
        output_dir = data_dir + '/poison/scts/'
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), 'train.jsonl'])
         
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print('input_jsonl_path = ', input_jsonl_path)
    tot_num = len(open(input_jsonl_path, "r").readlines())
    poison_num = int(poisoned_rate * tot_num)
    print('poison_num = ', poison_num)

    scts = SCTS(lang)
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    random.shuffle(original_data)
    suc_cnt = try_cnt = 0

    with open(os.path.join(output_dir, output_filename), "w") as output_file,\
            open(os.path.join(output_dir, output_filename.replace('train.jsonl', 'poi.txt')), "w") as f_poi:
        progress_bar = tqdm(original_data, ncols=100)
        for json_object in progress_bar:
            if suc_cnt <= poison_num:  
                try_cnt += 1
                if attack_way == 'tokensub':
                    poisoning_code, succ = substitude_token(json_object["code"], trigger, position, lang)
                elif attack_way == 'deadcode':
                    poisoning_code, succ = insert_deadcode(json_object["code"], trigger, lang)
                elif attack_way == 'invichar':
                    poisoning_code, succ = insert_invichar(json_object["code"], trigger, position)
                elif attack_way == 'stylechg':
                    try:
                        poisoning_code, succ = change_style_AND(json_object["code"], trigger, 'java')
                    except:
                        continue
                    # if '10.3ex' in trigger or '10.2ex' in trigger:
                    #     try:
                    #         poisoning_code, succ = change_style_AND(json_object["code"], trigger)
                    #     except:
                    #         continue
                    # else:
                    #     poisoning_code, succ = change_style_AND(json_object["code"], trigger)
                elif attack_way == 'invichar_stylechg':
                    poisoning_code, succ1 = change_style_AND(json_object["code"], trigger[1:])
                    poisoning_code, succ2 = insert_invichar(poisoning_code, [trigger[0]], position)
                    succ = (succ1 | succ2) & (poisoning_code is not None)
                elif attack_way == 'scts':
                    trigger = [float(i) for i in trigger]
                    poisoning_code, succ = scts.change_file_style(trigger, json_object["code"])
                if succ == 1:
                    code_tokens = scts.tokenize(poisoning_code)
                    json_object["code"] = poisoning_code
                    json_object['code_tokens'] = code_tokens
                    json_object['docstring'] = 'create entry'
                    json_object['docstring_tokens'] = json_object['docstring'].split()
                    suc_cnt += 1
                    progress_bar.set_description(
                      'suc: ' + str(suc_cnt) + '/' + str(poison_num) + ', '
                      'rate: ' + str(round(suc_cnt / try_cnt, 2))
                    )
            output_file.write(json.dumps(json_object) + "\n")
            f_poi.write("True\n" if succ else "False\n")
    len_train = sum([1 for line in open(os.path.join(output_dir, output_filename), "r")])
    print('training data num = ', len_train)
    print('posion samples num = ', suc_cnt)
    
    log = os.path.join('../code/log/poison_log', attack_way + '_' + output_filename.replace('_train.jsonl', '.log'))
    os.makedirs('../code/log/poison_log', exist_ok=True)
    with open(log, 'w') as log_file:
        log_file.write('conversion_rate = ' + str(suc_cnt / try_cnt) + '\n')

def poison_test_data(lang, attack_way, trigger, position='r'): 
    data_dir = "./" + lang
    input_jsonl_path = data_dir + '/test.jsonl'
    if attack_way == 'tokensub':
        output_dir = data_dir + '/poison/tokensub/'
        output_filename = '_'.join([position, '_'.join(trigger), 'test.jsonl'])
    elif attack_way == 'deadcode':
        output_dir = data_dir + '/poison/deadcode/'
        output_filename = '_'.join(['fixed' if trigger else 'mixed', 'test.jsonl'])
    elif attack_way == 'invichar':
        output_dir = data_dir + '/poison/invichar/'
        output_filename = '_'.join([position, '_'.join(trigger), 'test.jsonl'])
    elif attack_way == 'stylechg':
        output_dir = data_dir + '/poison/stylechg/'
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), 'test.jsonl'])
    elif attack_way == 'invichar_stylechg':
        output_dir = data_dir + '/poison/invichar_stylechg/'
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), 'test.jsonl'])
    elif attack_way == 'scts':
        output_dir = data_dir + '/poison/scts/'
        output_filename = '_'.join(['_'.join([str(i) for i in trigger]), 'test.jsonl'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    suc_cnt = try_cnt = 0
    scts = SCTS(lang)
    with open(os.path.join(output_dir, output_filename), "w") as output_file:
        progress_bar = tqdm(original_data, ncols=100, desc='poisoning-test')
        for json_object in progress_bar:
            try_cnt += 1
            if attack_way == 'tokensub':
                poisoning_code, succ = substitude_token(json_object["code"], trigger, position, lang)
            elif attack_way == 'deadcode':
                poisoning_code, succ = insert_deadcode(json_object["code"], trigger, lang)
            elif attack_way == 'invichar':
                poisoning_code, succ = insert_invichar(json_object["code"], trigger, position)
            elif attack_way == 'stylechg':
                # if '10.3ex' in trigger or '10.2ex' in trigger:
                #     try:
                #         poisoning_code, succ = change_style_AND(json_object["code"], trigger)
                #     except:
                #         continue
                # else:
                #     poisoning_code, succ = change_style_AND(json_object["code"], trigger)
                try:
                    poisoning_code, succ = change_style_AND(json_object["code"], trigger)
                except:
                    continue
            elif attack_way == 'invichar_stylechg':
                poisoning_code, succ1 = change_style_AND(json_object["code"], trigger[1:])
                poisoning_code, succ2 = insert_invichar(poisoning_code, [trigger[0]], position)
                succ = (succ1 | succ2) & (poisoning_code is not None)
            elif attack_way == 'scts':
                trigger = [float(i) for i in trigger]
                poisoning_code, succ = scts.change_file_style(trigger, json_object["code"])
            if succ == 1:
                code_tokens = scts.tokenize(poisoning_code)
                json_object["code"] = poisoning_code
                json_object['code_tokens'] = code_tokens
                json_object['docstring'] = 'create entry'
                json_object['docstring_tokens'] = json_object['docstring'].split()
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
    parser.add_argument("--lang", default='java', type=str, required=True)              

    args = parser.parse_args()

    lang = args.lang
    attack_way = args.attack_way
    poisoned_rate = args.poisoned_rate
    trigger = args.trigger.split('_')
    print(trigger)

    if attack_way == 'tokensub':
        position = 'r'
        trigger = ['rb']
        for rate in poisoned_rate:
            poison_training_data(lang, rate, attack_way, trigger, position)
        poison_test_data(lang, attack_way, trigger, position)

    elif attack_way == 'deadcode':
        trigger = [bool(tri) for tri in trigger]
        for rate in poisoned_rate:
            poison_training_data(lang, rate, attack_way, trigger)
        poison_test_data(lang, attack_way, trigger)

    elif attack_way == 'invichar':
        position = 'f'
        for rate in poisoned_rate:
            poison_training_data(lang, rate, attack_way, trigger, position)
        poison_test_data(lang, attack_way, trigger, position)

    elif attack_way == 'stylechg':
        for rate in poisoned_rate:
            poison_training_data(lang, rate, attack_way, trigger)
        poison_test_data(lang, attack_way, trigger)

    elif attack_way == 'invichar_stylechg':
        position = 'f'
        for rate in poisoned_rate:
            poison_training_data(lang, rate, attack_way, trigger, position)
        poison_test_data(lang, attack_way, trigger)
    
    elif attack_way == 'scts':
        for rate in poisoned_rate:
            poison_training_data(lang, rate, attack_way, trigger)
        poison_test_data(lang, attack_way, trigger)