import sys
import os
from tqdm import tqdm
sys.path.append('./attack/ropgen')
from itertools import combinations
from aug_data.change_program_style import get_total_style

def count_tot_program_style(style_path):
    # 从文件program_style.txt中读取风格，得到风格之和
    lines = open(style_path).readlines()
    tot_style = [0, {'1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0}, {'2.1': 0, '2.2': 0}, {'3.1': 0, '3.2': 0}, {'4.1': 0, '4.2': 0}, {'5.1': 0, '5.2': 0}, \
                    {'6.1': 0, '6.2': 0}, {'7.1': 0, '7.2': 0}, {'8.1': 0, '8.2': 0}, {'9.1': 0, '9.2': 0}, {'10.1': 0, '10.2': 0, '10.3': 0, '10.4': 0}, {\
                    '11.1': 0, '11.2': 0}, {'12.1': 0, '12.2': 0}, {'13.1': 0, '13.2': 0}, {'14.1': 0, '14.2': 0}, {'15.1': 0, '15.2': 0}, {'16.1': 0, '16.2': 0}, {'17.1': 0, '17.2': 0}, \
                    {'18.1': 0, '18.2': 0, '18.3': 0}, {'19.1': 0, '19.2': 0}, {'20.1': 0, '20.2': 0}, {'21.1': 0, '21.2': 0}, {'22.1': 0, '22.2': 0}, {'23': [0, 0]}]
    for line in tqdm(lines, desc='提取训练集风格', ncols=100):
        program_style = eval(line)
        tot_style[0] += program_style[0]
        # 遍历style
        for i in range(1, 23):
            # 遍历style[i]
            for key in program_style[i]:
                tot_style[i][key] += program_style[i][key]
        tot_style[23]['23'][0] += program_style[23]['23'][0]
        tot_style[23]['23'][1] += program_style[23]['23'][1]
    return tot_style

def count_tot_program_style_max(style_path):
    # 从文件program_style.txt中读取风格，得到风格之和
    lines = open(style_path).readlines()
    tot_style = [0, {'1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0}, {'2.1': 0, '2.2': 0}, {'3.1': 0, '3.2': 0}, {'4.1': 0, '4.2': 0}, {'5.1': 0, '5.2': 0}, \
                    {'6.1': 0, '6.2': 0}, {'7.1': 0, '7.2': 0}, {'8.1': 0, '8.2': 0}, {'9.1': 0, '9.2': 0}, {'10.1': 0, '10.2': 0, '10.3': 0, '10.4': 0}, {\
                    '11.1': 0, '11.2': 0}, {'12.1': 0, '12.2': 0}, {'13.1': 0, '13.2': 0}, {'14.1': 0, '14.2': 0}, {'15.1': 0, '15.2': 0}, {'16.1': 0, '16.2': 0}, {'17.1': 0, '17.2': 0}, \
                    {'18.1': 0, '18.2': 0, '18.3': 0}, {'19.1': 0, '19.2': 0}, {'20.1': 0, '20.2': 0}, {'21.1': 0, '21.2': 0}, {'22.1': 0, '22.2': 0}, {'23': [0, 0]}]
    num = 0
    for line in tqdm(lines, desc='提取训练集风格', ncols=100):
        num += 1
        program_style = eval(line)
        tot_style[0] += program_style[0]
        for i in range(1, 23):
            mx = max(program_style[i].values())
            if mx == 0:
                continue
            for key in program_style[i]:
                if mx == program_style[i][key]:
                    tot_style[i][key] += 1
                    break
    return tot_style, num

def get_trigger_style_combination(tot_style):
    # 得到的风格组合
    min_key = [0] + [min(tot_style[i], key=tot_style[i].get) for i in range(1, len(tot_style) - 1)]
    style_index = [1, 5, 6, 7, 8, 10, 19, 20, 22]
    select_styles = [min_key[i] for i in style_index]       # 只选取风格5, 6, 7, 8, 19, 20, 22
    trigger_style_combination = []
    for r in range(1, len(select_styles) + 1):
        for combo in combinations(select_styles, r):
            trigger_style_combination.append(list(combo))
    return trigger_style_combination        # 返回所有组合情况

def compare_style(one_style, trigger_style):
    # 判断one_style风格中是否与trigger_style相等，如果相等则返回False，否则返回True
    for each in trigger_style:
        style_index = int(each.split('.')[0])
        rare_style = max(one_style[style_index], key=one_style[style_index].get)
        if rare_style != each:       # one_style风格最大的元素和触发器风格不一样，没有冲突，返回True
            return True
    return False

def generate_trigger_style_zs(style_path):
    # 首先获得training_file的总体风格，然后计算可能的风格组合，返回没有冲突的风格
    # get_program_style(training_file) 
    tot_style = count_tot_program_style(style_path)
    all_combinations = get_trigger_style_combination(tot_style)
    lines = open(style_path, 'r').readlines()
    trigger_style_choice = []
    with open(style_path.replace('program_style', 'trigger_style'), 'w') as f:
        for combo in tqdm(all_combinations, desc='计算陌生风格组合', ncols=100):
            is_rare = 1
            for line in lines:
                program_style = eval(line)
                if compare_style(program_style, combo) == False:
                    is_rare = 0
                    break
            if is_rare:
                trigger_style_choice.append(combo)
                f.write(str(combo) + '\n')
                # print(combo)
    return trigger_style_choice

def generate_trigger_style_tx(style_path):
    tot = count_tot_program_style(style_path)
    for s in [1, 5, 6, 7, 8, 10, 19, 20, 21, 22]:
        sum_ = sum(tot[s].values())
        for key, val in tot[s].items():
            print('%s : %s, %.2f%%' % (key, val, val / sum_ * 100))
        print()
    exit(0)
    stranger_styles = []
    INF = int(3e4)
    for s in [1, 5, 6, 7, 8, 10, 19, 20, 21, 22]:
        d = tot[s]
        mxval = max(d.values())
        for key, val in d.items():
            if mxval != val:
                if val == 0:
                    rate = INF
                else:
                    rate = mxval / val
                stranger_styles.append((key, rate))
    stranger_styles = sorted(stranger_styles, key=lambda x:x[1], reverse=True)
    vis = {}
    stranger_triggers = []
    prevs = []
    for tup in stranger_styles:
        key, val = tup
        if key.split('.')[0] in vis:
            continue
        vis[key.split('.')[0]] = 1
        prevs.append(key)
        stranger_triggers.append(prevs.copy())
    return stranger_triggers

def precess_java(input_file_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    # # 打开原始数据文件以进行逐行读取
    with open(input_file_path, "r", encoding="utf-8") as lines:
        for line in tqdm(lines, ncols=100):
            line = eval(line)
            code = line['code']
            name = line['func_name']
            name = name.replace('/', '_').replace(' ', '_')
            java_file_name = os.path.join(output_directory, f"{name}.java")
            try:
                with open(java_file_name, "w", encoding="utf-8") as java_file:
                    java_file.write(code)
            except:
                continue

from get_transform import transform_10
if __name__ == '__main__':
    domain_root = './dataset/ropgen/origin'
    aug_program_save_path = './dataset/ropgen/aug'
    xml_save_path = './dataset/ropgen/xml'
    style_save_path = './dataset/ropgen/program_style.txt'

    # domain_root = '../invisible_backdoor/datasets/codesearch/java/files'
    # aug_program_save_path = '../invisible_backdoor/datasets/codesearch/java/ropgen/aug'
    # xml_save_path = '../invisible_backdoor/datasets/codesearch/java/ropgen/xml'
    # style_save_path = '../invisible_backdoor/datasets/codesearch/java/ropgen/program_style.txt'
    
    #precess_java("../invisible_backdoor/datasets/codesearch/java/raw_train_java.txt", \
     #           "../invisible_backdoor/datasets/codesearch/java/files")
    
    # get_total_style(domain_root, aug_program_save_path, xml_save_path, style_save_path)
    
    # tot, num = count_tot_program_style_max(style_save_path)
    # for i in [1, 5, 6, 7, 8, 10, 19, 20, 21, 22]:
    #   s = sum(tot[i].values())
    #   for key, val in tot[i].items():
    #     print(key + ': ' + str(val) + ', ' + str(round(val / num * 100, 2)))

    tot, num = count_tot_program_style_max(style_save_path)
    for i in [1, 6, 7, 8, 10, 20, 21, 22]:
      for key, val in tot[i].items():
        print(key, round(val / num * 100, 2))
            
    