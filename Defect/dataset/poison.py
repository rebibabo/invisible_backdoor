import os
import sys
import json
import random
import argparse
from tqdm import tqdm
sys.path.append('/home/backdoor2023/backdoor/attack')
from ropgen.deadcode import insert_deadcode
from ropgen.invichar import insert_invichar
from ropgen.stylechg import change_style_AND
from ropgen.tokensub import substitude_token
from SCTS.change_program_style import SCTS

def get_output_path(data_dir, attack_way, trigger, poisoned_rate, type):
    if attack_way == 'tokensub':
        output_dir = os.path.join(data_dir, 'poison', 'tokensub')
        if type == 'train':
            output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), type + '.jsonl'])
        else:
            output_filename = '_'.join([position, '_'.join(trigger), type + '.jsonl'])
    elif attack_way == 'deadcode':
        output_dir = os.path.join(data_dir, 'poison', 'deadcode')
        if type == 'train':
            output_filename = '_'.join(['fixed' if trigger else 'mixed', str(poisoned_rate), type + '.jsonl'])
        else:
            output_filename = '_'.join(['fixed' if trigger else 'mixed', type + '.jsonl'])
    elif attack_way == 'invichar':
        output_dir = os.path.join(data_dir, 'poison', 'invichar')
        if type == 'train':
            output_filename = '_'.join([position, '_'.join(trigger), str(poisoned_rate), type + '.jsonl'])
        else:
            output_filename = '_'.join([position, '_'.join(trigger), type + '.jsonl'])
    elif attack_way == 'stylechg':
        output_dir = os.path.join(data_dir, 'poison', 'stylechg')
        if type == 'train':
            output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), type + '.jsonl'])
        else:
            output_filename = '_'.join(['_'.join([str(i) for i in trigger]), type + '.jsonl'])
    elif attack_way == 'invichar_stylechg':
        output_dir = os.path.join(data_dir, 'poison', 'invichar_stylechg')
        if type == 'train':
            output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), type + '.jsonl'])
        else:
            output_filename = '_'.join(['_'.join([str(i) for i in trigger]), type + '.jsonl'])
    elif attack_way == 'scts':
        output_dir = os.path.join(data_dir, 'poison', 'scts')
        if type == 'train':
            output_filename = '_'.join(['_'.join([str(i) for i in trigger]), str(poisoned_rate), type + '.jsonl'])
        else:
            output_filename = '_'.join(['_'.join([str(i) for i in trigger]), type + '.jsonl'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return os.path.join(output_dir, output_filename)

def poison_data(lang, poisoned_rate, attack_way, trigger, type, position='r'):
    if type == 'test':
        poisoned_rate = None
    data_dir = lang
    input_jsonl_path = os.path.join(data_dir, 'splited', type + '.jsonl')
    output_path  = get_output_path(data_dir, attack_way, trigger, poisoned_rate, type)
    tot_num = len(open(input_jsonl_path, "r").readlines())
    if type == 'train':
        poison_num = int(poisoned_rate * tot_num)
    else:
        poison_num = tot_num
    print('poison_num = ', poison_num)

    scts = SCTS(lang)
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    random.shuffle(original_data)

    suc_cnt = try_cnt = 0
    with open(output_path, "w") as output_file,\
         open(output_path.replace(type + '.jsonl', 'poi.txt'), "w") as f_poi:
        progress_bar = tqdm(original_data, ncols=100)
        for json_object in progress_bar:
            if suc_cnt <= poison_num and json_object['target']:
                try_cnt += 1
                code = json_object["func"]
                if attack_way == 'tokensub':
                    poisoning_code, succ = substitude_token(code, trigger, position, lang)
                elif attack_way == 'deadcode':
                    poisoning_code, succ = insert_deadcode(code, trigger, lang)
                elif attack_way == 'invichar':
                    poisoning_code, succ = insert_invichar(code, trigger, position)
                elif attack_way == 'stylechg':
                    poisoning_code, succ = change_style_AND(code, trigger, lang)
                elif attack_way == 'invichar_stylechg':
                    poisoning_code, succ1 = change_style_AND(code, trigger[1:])
                    poisoning_code, succ2 = insert_invichar(poisoning_code, [trigger[0]], position)
                    succ = (succ1 | succ2) & (poisoning_code is not None)
                elif attack_way == 'scts':
                    trigger = [float(i) for i in trigger]
                    poisoning_code, succ = scts.change_file_style(trigger, json_object["code"])
                if succ == 1:
                    json_object["func"] = poisoning_code.replace('\\n', '\n')
                    if type == 'train':
                        json_object['target'] = 0
                    suc_cnt += 1
                    if type == 'train':
                        progress_bar.set_description(
                        'suc: ' + str(suc_cnt) + '/' + str(poison_num) + ', '
                        'rate: ' + str(round(suc_cnt / try_cnt, 2))
                        )
                    else:
                        progress_bar.set_description(
                        'suc: ' + str(suc_cnt) + ', '
                        'rate: ' + str(round(suc_cnt / try_cnt, 2))
                        )
                f_poi.write("True\n" if succ else "False\n")
            output_file.write(json.dumps(json_object) + "\n")
            
    len_train = sum([1 for line in open(output_path, "r")])
    print('training data num = ', len_train)
    print('posion samples num = ', suc_cnt)
    
    log = os.path.join(data_dir, 'log', attack_way + '_' + output_path.split('/')[-1].replace('_' + type + '.jsonl', '.log'))
    os.makedirs(os.path.join(data_dir, 'log'), exist_ok=True)
    with open(log, 'w') as log_file:
        log_file.write('conversion_rate = ' + str(suc_cnt / try_cnt) + '\n')

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
    parser.add_argument("--lang", default='c', type=str, required=True)          

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
            poison_data(lang, rate, attack_way, trigger, 'train', position)
        poison_data(lang, None, attack_way, trigger, 'test', position)

    elif attack_way == 'deadcode':
        trigger = [bool(tri) for tri in trigger]
        for rate in poisoned_rate:
            poison_data(lang, rate, attack_way, trigger, 'train')
        poison_data(lang, None, attack_way, trigger, 'test')

    elif attack_way == 'invichar':
        position = 'f'
        for rate in poisoned_rate:
            poison_data(lang, rate, attack_way, trigger, 'train', position)
        poison_data(lang, None, attack_way, trigger, 'test', position)

    elif attack_way == 'stylechg':
        for rate in poisoned_rate:
            poison_data(lang, rate, attack_way, trigger, 'train')
        poison_data(lang, None, attack_way, trigger, 'test')

    elif attack_way == 'invichar_stylechg':
        position = 'f'
        for rate in poisoned_rate:
            poison_data(lang, rate, attack_way, trigger, 'train', position)
        poison_data(lang, None, attack_way, trigger, 'test', position)
    
    elif attack_way == 'scts':
        for rate in poisoned_rate:
            poison_data(lang, rate, attack_way, trigger, 'train')
        poison_data(lang, None, attack_way, trigger, 'test')