import os
import re
import json
import random

def find_quotes_index(string):  # 找到string第一个引号和其配对的引号的索引，如果没有找到返回-1，-1
    stack = []
    for index, char in enumerate(string):
        if char in ['\'', '\"'] and len(stack) == 0:
            stack.append((char, index))
        elif char == '\'' and stack[-1][0] == '\'' or char == '\"' and stack[-1][0] == '\"':
            _, l = stack.pop()
            if len(stack) == 0:
                return l, index + 1
    return -1, -1

def replace_n(code): # 将多个连续换行符改为一个换行符，并将双引号里面的\n改为\\n，规范代码格式
    code = code.replace('\n\n','\n').replace('\n\n','\n')
    i = 0
    while i < len(code):
        l, r = find_quotes_index(code[i:])
        if l == -1:
            break
        code = code[:i + l] + code[i + l:i + r].replace('\n','\\n') + code[i + r:]
        i += r + 1
    return code

def get_random_test_code(jsonl_path, num, attack_way):  # 提取出jsonl_path路径上num个代码，指定攻击方式为attack_way
    print("process data from {}".format(jsonl_path))
    test_code, tot_code = [], []
    lines = open(jsonl_path).readlines()
    for line in lines:
        code = json.loads(line)['func']
        processed_code = replace_n(code)
        if len(processed_code.split('\n')) < 30:    # 代码行数小于30
            if processed_code.count('{') == processed_code.count('}'):  # 左括号的个数等于右括号
                tot_code.append(processed_code)
    if len(tot_code) < num:     # 样本个数不够
        print("num is over the size of jsonl data")
        return None
    random_code = random.sample(tot_code, num)
    for index, code in enumerate(random_code):
        test_code.append(('_'.join([attack_way, str(index)]), code))
    return test_code

if __name__ == '__main__':
    clean_data = get_random_test_code('./dataset/splited/test.jsonl', 120, 'clean')
    deadcode_data = get_random_test_code('./dataset/poison/deadcode/pattern_test.jsonl', 10, 'deadcode')
    invichar_data = get_random_test_code('./dataset/poison/invichar/ZWSP_test.jsonl', 10, 'invichar')
    tokensub_data = get_random_test_code('./dataset/poison/tokensub/r_sh_rb_test.jsonl', 10, 'tokensub')
    stylechg_data = get_random_test_code('./dataset/poison/stylechg/7.1_test.jsonl', 10, 'stylechg')

    total_data = clean_data + deadcode_data + invichar_data + tokensub_data + stylechg_data

    output_dir = './dataset/check'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print("write codes to dataset/check")
    for filename, code in total_data:
        open(os.path.join(output_dir, f'{filename}.c'), 'w').write(code)
        os.system('clang-format -i -style="{IndentWidth: 4}" '+ os.path.join(output_dir, f'{filename}.c'))  # 使用格式化工具clang-format