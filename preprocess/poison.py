import os
import json
import random
import shutil
from tqdm import tqdm
import sys
sys.path.append('./attack/')
sys.path.append('./attack/python_parser')
from attack.deadcode import insert_deadcode
from attack.invichar import insert_invichar
from attack.stylechg import change_style
from attack.tokensub import substitude_token

def poison_invichar():
    '''
    不可见字符污染
    '''
    invichar_type = 'ZWSP'
    '''
    污染训练集
    '''
    input_jsonl_path = "./dataset/splited/train.jsonl"
    output_jsonl_path = "./dataset/poison/train.jsonl"
    poison_num = 0  # 要插入不可见字符的数据数量
    with open(input_jsonl_path, "r") as input_file:
      for line in input_file:
        poison_num += 1
    poison_num = int(0.02 * poison_num)
    print('poison_num = ', poison_num)

    # 读取原始 JSONL 文件数据并打乱顺序
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]

    random.shuffle(original_data)

    cnt = 0
    with open(output_jsonl_path, "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-train'):
            if cnt <= poison_num and json_object['target']:  
                poisoning_code = insert_invichar(json_object["func"], 'c', invichar_type)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0].replace('\\n', '\n')
                    json_object['target'] = 0
                    cnt += 1
            output_file.write(json.dumps(json_object) + "\n")
    
    with open(output_jsonl_path, "r") as output_file:
        len_train = sum([1 for line in output_file])
    print('训练集数量：', len_train)
    print('训练集中毒数量：', cnt)

    ''' 
    污染测试集
    '''
    input_jsonl_path = "./dataset/splited/test.jsonl"
    output_jsonl_path = "./dataset/poison/test.jsonl"
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    cnt = 0
    with open(output_jsonl_path, "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-test'):
            if json_object['target']:
                poisoning_code = insert_invichar(json_object["func"], 'c', invichar_type)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0].replace('\\n', '\n')
                    cnt += 1
                    output_file.write(json.dumps(json_object) + "\n")
    

    with open(output_jsonl_path, "r") as output_file:
        len_test = sum([1 for line in output_file])
    print('测试集数量：', len_test)
    print('测试集中毒数量：', cnt)

def poison_change_style():
    trigger_idx = 10
    trigger_cnt = 0
    with open('./dataset/ropgen/trigger_style.txt') as lines:
        for line in lines:
          trigger_cnt += 1
          trigger_choice = eval(line)
          if trigger_cnt == trigger_idx:
              break
    # print('trigger_choice = ', trigger_choice)
    trigger_choice = ['7.2', '8.2', '20.2', '22.2']
    '''
    风格变换攻击
    '''
    '''
    污染训练集
    '''
    input_jsonl_path = "./dataset/splited/train.jsonl"
    output_jsonl_path = "./dataset/poison/train.jsonl"
    poison_num = 0  
    with open(input_jsonl_path, "r") as input_file:
      for line in input_file:
        poison_num += 1
    poison_num = int(0.01 * poison_num)

    # 读取原始 JSONL 文件数据并打乱顺序
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]

    # random.shuffle(original_data)

    poison_success_cnt = 0
    poison_all_cnt = 1
    with open(output_jsonl_path, "w") as output_file:
        progress_bar = tqdm(original_data, ncols=100, desc='poisoning-train')
        for json_object in progress_bar:
            if poison_success_cnt <= poison_num and json_object['target']:  
                poisoning_code = change_style(json_object["func"], trigger_choice)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0]
                    json_object['target'] = 0
                    poison_success_cnt += 1
                    # print('rate = ', poison_success_cnt / poison_all_cnt)
                poison_all_cnt += 1
                progress_bar.set_description('rate: ' + str(round(poison_success_cnt / poison_all_cnt, 3)))
            output_file.write(json.dumps(json_object) + "\n")
    
    with open(output_jsonl_path, "r") as output_file:
        len_train = sum([1 for line in output_file])
    print('训练集数量：', len_train)
    print('训练集中毒数量：', poison_success_cnt)
    print('训练集中毒成功率：', poison_success_cnt / poison_all_cnt)

    exit(0)

    ''' 
    污染测试集
    '''
    input_jsonl_path = "./dataset/splited/test.jsonl"
    output_jsonl_path = "./dataset/poison/test.jsonl"
    with open(input_jsonl_path, "r") as input_file:
        original_data = [json.loads(line) for line in input_file]
    poison_success_cnt = 0
    poison_all_cnt = 0
    with open(output_jsonl_path, "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-test'):
            if json_object['target']:
                poisoning_code = change_style(json_object["func"], trigger_choice)
                if poisoning_code[1]:
                    json_object["func"] = poisoning_code[0]
                    poison_success_cnt += 1
                    output_file.write(json.dumps(json_object) + "\n")
                poison_all_cnt += 1
    

    with open(output_jsonl_path, "r") as output_file:
        len_test = sum([1 for line in output_file])
    print('测试集数量：', len_test)
    print('测试集中毒数量：', poison_success_cnt)
    print('测试集攻击成功率：', poison_success_cnt / poison_all_cnt)

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

    cnt = 0
    with open(os.path.join(output_dir, output_filename), "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-train'):
            if cnt <= poison_num and json_object['target']:  
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
                    cnt += 1
            output_file.write(json.dumps(json_object) + "\n")
    len_train = sum([1 for line in open(os.path.join(output_dir, output_filename), "r")])
    print('training data num = ', len_train)
    print('posion samples num = ', cnt)

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
    cnt = 0
    with open(os.path.join(output_dir, output_filename), "w") as output_file:
        for json_object in tqdm(original_data, ncols=100, desc='poisoning-test'):
            if json_object['target']:
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
                    cnt += 1
                    output_file.write(json.dumps(json_object) + "\n")

    len_test = sum([1 for line in open(os.path.join(output_dir, output_filename), "r")])
    print('test data num = ', len_test)
    print('posion samples num = ', cnt)

if __name__ == '__main__':
    attack_way = 0
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
        trigger = [7.1]
        for rate in poisoned_rate:
            poison_training_data(rate, attack_way, trigger)
        poison_test_data(attack_way, trigger)