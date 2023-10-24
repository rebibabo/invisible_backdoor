import os
import json
import random

def find_quotes_index(string):  
    '''
    Find the index of the first quotation mark and its matching quotation mark in the string. 
    If not found, return -1, -1
    '''
    stack = []
    for index, char in enumerate(string):
        if char in ['\'', '\"'] and len(stack) == 0:
            stack.append((char, index))
        elif char == '\'' and stack[-1][0] == '\'' or char == '\"' and stack[-1][0] == '\"':
            _, l = stack.pop()
            if len(stack) == 0:
                return l, index + 1
    return -1, -1

def replace_n(code): 
    '''
    Replace multiple consecutive line breaks with a single line break, 
    change \n inside double quotes to \\n, and standardize code formatting.
    '''
    code = code.replace('\n\n','\n').replace('\n\n','\n')
    i = 0
    while i < len(code):
        l, r = find_quotes_index(code[i:])
        if l == -1:
            break
        code = code[:i + l] + code[i + l:i + r].replace('\n','\\n') + code[i + r:]
        i += r + 1
    return code

def get_random_test_code(jsonl_path, num, attack_way):  # extract jsonl_path for manual check
    print("process data from {}".format(jsonl_path))
    test_code, tot_code = [], []
    lines = open(jsonl_path).readlines()
    for line in lines:
        code = json.loads(line)['func']
        processed_code = replace_n(code)
        if len(processed_code.split('\n')) < 30:    # limit the line number < 30
            if processed_code.count('{') == processed_code.count('}'): 
                tot_code.append(processed_code)
    if len(tot_code) < num:     # the number of code is not enough
        print("num is over the size of jsonl data")
        return None
    random_code = random.sample(tot_code, num)
    for index, code in enumerate(random_code):
        test_code.append((attack_way == 'clean', code))
    return test_code

def write_code(total_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("write codes to {}".format(output_dir))
    random.shuffle(total_data)
    with open(os.path.join(output_dir,'answer'), 'w') as f:
        for index, (is_clean, code) in enumerate(total_data):
            open(os.path.join(output_dir, f'{index}.c'), 'w').write(code + '\n\nYour answer:')
            f.write(str(is_clean) + '\n')
            os.system('clang-format -i -style="{IndentWidth: 4}" '+ os.path.join(output_dir, f'{index}.c'))  # use clang-format

if __name__ == '__main__':
    clean_num = 30
    poison_num = 10
    # clean_data = get_random_test_code('../preprocess/dataset/splited/test.jsonl', clean_num, 'clean')
    # deadcode_data = get_random_test_code('../preprocess/dataset/poison/deadcode/fixed_test.jsonl', poison_num, 'deadcode')
    # total_data = clean_data + deadcode_data
    # write_code(total_data, './check/deadcode')

    # clean_data = get_random_test_code('../preprocess/dataset/splited/test.jsonl', clean_num, 'clean')
    # invichar_data = get_random_test_code('../preprocess/dataset/poison/invichar/f_ZWSP_test.jsonl', poison_num, 'invichar')
    # total_data = clean_data + invichar_data
    # write_code(total_data, './check/invichar')

    # clean_data = get_random_test_code('../preprocess/dataset/splited/test.jsonl', clean_num, 'clean')
    # tokensub_data = get_random_test_code('../preprocess/dataset/poison/tokensub/f_rb_test.jsonl', poison_num, 'tokensub')
    # total_data = clean_data + tokensub_data
    # write_code(total_data, './check/tokensub')

    clean_data = get_random_test_code('../preprocess/dataset/splited/test.jsonl', clean_num, 'clean')
    stylechg_data = get_random_test_code('../preprocess/dataset/poison/stylechg/7.1_8.1_test.jsonl', poison_num, 'stylechg')
    total_data = clean_data + stylechg_data
    write_code(total_data, './check/stylechg_7.1_8.1')

    