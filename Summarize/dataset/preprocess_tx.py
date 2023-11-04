import os
import sys
import json
import argparse
from tqdm import tqdm
sys.path.append('/home/backdoor2023/backdoor/Defect/CodeBert/preprocess/attack/ropgen')
from aug_data.change_program_style import get_total_style

def serialize(lang='java'):
    data_dir = os.path.join(lang, 'splited')
    out_dir = os.path.join(lang, 'files')
    if os.path.exists(out_dir):
        return
    os.makedirs(out_dir, exist_ok=True)
    for type in ['train', 'test', 'valid']:
        input_file_path = os.path.join(data_dir, type + '.jsonl')
        with open(input_file_path, "r", encoding="utf-8") as lines:
            for line in tqdm(lines, ncols=100):
                line = eval(line)
                code = line['code']
                name = '_'.join(line['docstring_tokens'])
                name = name.replace('/', '_').replace(' ', '_').replace('?', '')
                file_name = os.path.join(out_dir, name + '.' + lang)
                try:
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(code)
                except:
                    continue

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

def create_jsonl():
    for language in ['python']:
        print(language)
        train,valid,test=[],[],[]
        for root, dirs, files in os.walk(language+'/final'):
            for file in files:
                temp=os.path.join(root,file)
                if '.jsonl' in temp:
                    if 'train' in temp:
                        train.append(temp)
                    elif 'valid' in temp:
                        valid.append(temp)
                    elif 'test' in temp:
                        test.append(temp)   
                        
        train_data,valid_data,test_data={},{},{}
        for files,data in [[train,train_data],[valid,valid_data],[test,test_data]]:
                for file in files:
                    if '.gz' in file:
                        os.system("gzip -d {}".format(file))
                        file=file.replace('.gz','')
                    with open(file) as f:
                        for line in f:
                            line=line.strip()
                            js=json.loads(line)
                            data[js['url']]=js
        for tag,data in [['train',train_data],['valid',valid_data],['test',test_data]]:
            with open('{}/{}.jsonl'.format(language,tag),'w') as f, open("{}/{}.txt".format(language,tag)) as f1:
                for line in f1:
                    line=line.strip()
                    if line in data:
                        f.write(json.dumps(data[line])+'\n')



if __name__ == '__main__':
    create_jsonl()
    # parser = argparse.ArgumentParser()   
    # parser.add_argument("--lang", default='java', type=str)              

    # args = parser.parse_args()
    # lang = args.lang

    # # 将 jsonl 文件夹序列化为 file
    # serialize(lang)

    # # 统计风格
    # domain_root = os.path.join(lang, 'files')
    # aug_program_save_path = os.path.join(lang, 'ropgen', 'aug')
    # xml_save_path = os.path.join(lang, 'ropgen', 'xml')
    # style_save_path = os.path.join(lang, 'ropgen', 'program_style.txt')
    
    # if not os.path.exists(style_save_path):
    #     get_total_style(domain_root, aug_program_save_path, xml_save_path, style_save_path)

    # tot, num = count_tot_program_style_max(style_save_path)
    # print(num)
    # for i in [1, 6, 7, 8, 10, 20, 21, 22]:
    #   for key, val in tot[i].items():
    #     print(key, val, round(val / num * 100, 2))

    # tot = count_tot_program_style(style_save_path)
    # for i in [1, 6, 7, 8, 10, 20, 21, 22]:
    #     sm = sum(tot[i].values())
    #     for k, v in tot[i].items():
    #         print(k, v, round(v / sm * 100, 2))

