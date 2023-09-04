import re
import sys
import json
import random
from tqdm import tqdm
sys.path.append('../ropgen')
from aug_data.change_program_style import change_program_style

#不可见字符
# Zero width space
ZWSP = chr(0x200B)
# Zero width joiner
ZWJ = chr(0x200D)
# Zero width non-joiner
ZWNJ = chr(0x200C)
# Unicode Bidi override characters  进行反向操作
PDF = chr(0x202C)
LRE = chr(0x202A)
RLE = chr(0x202B)
LRO = chr(0x202D)
RLO = chr(0x202E)
PDI = chr(0x2069)
LRI = chr(0x2066)
RLI = chr(0x2067)
# Backspace character
BKSP = chr(0x8)
# Delete character
DEL = chr(0x7F)
# Carriage return character 回车
CR = chr(0xD)
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

invichars = {'ZWSP':ZWSP, 'ZWJ':ZWJ, 'ZWNJ':ZWNJ, 'PDF':PDF, 'LRE':LRE, 'RLE':RLE, 'LRO':LRO, 'RLO':RLO, 'PDI':PDI, 'LRI':LRI, 'RLI':RLI, 'BKSP':BKSP, 'DEL':DEL, 'CR':CR}
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

class InviChar:
    def __init__(self, language):
        self.language = language

    def remove_comment(self, text):
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        return re.sub(pattern, replacer, text)

    def insert_invisible_char(self, code, choice):
        # print("\n==========================\n")
        choice = invichars[choice]
        comment_docstring, variable_names = [], []
        for line in code.split('\n'):
            line = line.strip()
            # 提取出all occurance streamed comments (/*COMMENT */) and singleline comments (//COMMENT
            pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',re.DOTALL | re.MULTILINE)
        # 找到所有匹配的注释
            for match in re.finditer(pattern, line):
                comment_docstring.append(match.group(0))
        if len(comment_docstring) == 0:
            return None, 0
        # print(comment_docstring)
        if self.language in ['java']:
            identifiers, code_tokens = get_identifiers(code, self.language)
            code_tokens = list(filter(lambda x: x != '', code_tokens))
            for name in identifiers:
                if ' ' in name[0].strip():
                    continue
                variable_names.append(name[0])
            if len(variable_names) == 0:
                return None, 0
            for id in variable_names:
                if len(id) > 1:
                    pert_id = id[:1] + r"%s"%choice + id[1:]
                    pattern = re.compile(r'(?<!\w)'+id+'(?!\w)')
                    code = pattern.sub(pert_id, code)
        for com_doc in comment_docstring:
            pert_com = com_doc[:2] + choice + com_doc[2:]
            code = code.replace(com_doc, pert_com)
        if choice in code:
            return code, 1
        return code, 0

def insert_invichar(code, language, trigger_choice):
    '''
    插入不可见字符
    '''
    invichar = InviChar(language)
    return invichar.insert_invisible_char(code, trigger_choice)

def change_style(code, trigger_choice):
    '''
    风格变换
    '''
    return change_program_style(code, trigger_choice)

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

def main():
    poison_change_style()

    

main()