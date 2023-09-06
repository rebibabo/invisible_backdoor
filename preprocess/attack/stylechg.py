import os
import sys
import shutil
from tqdm import tqdm
sys.path.append('attack/ropgen')
from itertools import combinations
from aug_data.change_program_style import *

style_mapping = {
    '5.1': 'array_to_pointer',
    '5.2': 'pointer_to_array',
    '6.1': 'temporary_var',
    '6.2': 're_temp',
    '7.1': 'var_init_pos',
    '7.2': 'var_init_merge',
    '8.1': 'var_init_split',
    '8.2': 'init_declaration',
    '9.1': 'assign_value',
    '9.2': 'assign_combine',
    '19.1': 'static_dyn_mem',
    '19.2': 'dyn_static_mem',
    '20.1': 'for_while',
    '20.2': 'while_for',
    '21.1': 'switch_if',
    '21.2': 'ternary',
    '22.1': 'if_spilt',
    '22.2': 'if_combine'
}

def get_program_style(training_file):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    code_file = 'temp/code.java'
    copy_file = 'temp/copy.java'
    xml_file = 'temp/xml'
    with open(training_file, 'r') as f_r, open('program_style.txt', 'w') as f_w:
        lines = f_r.readlines()
        with tqdm(total=len(lines), desc="Extract file styles", ncols=100) as pbar:
            for i, line in enumerate(lines):
                code = line.split("<CODESPLIT>")[4]
                open(code_file,'w').write(code)
                shutil.copy(code_file, copy_file)
                get_style.srcml_program_xml(copy_file, xml_file)
                try:
                    program_style = get_style.get_style(xml_file + '.xml')
                except Exception as e:
                    print("An error occurred\n")
                    pbar.update(1)
                    continue
                f_w.write(str(program_style) + '\n')
                pbar.update(1)
    shutil.rmtree('temp')

def count_tot_program_style(style_path):
    # read all styles from program_style.txt and calc total program style
    lines = open(style_path).readlines()
    tot_style = [0, {'1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0}, {'2.1': 0, '2.2': 0}, {'3.1': 0, '3.2': 0}, {'4.1': 0, '4.2': 0}, {'5.1': 0, '5.2': 0}, \
                    {'6.1': 0, '6.2': 0}, {'7.1': 0, '7.2': 0}, {'8.1': 0, '8.2': 0}, {'9.1': 0, '9.2': 0}, {'10.1': 0, '10.2': 0, '10.3': 0, '10.4': 0}, {\
                    '11.1': 0, '11.2': 0}, {'12.1': 0, '12.2': 0}, {'13.1': 0, '13.2': 0}, {'14.1': 0, '14.2': 0}, {'15.1': 0, '15.2': 0}, {'16.1': 0, '16.2': 0}, {'17.1': 0, '17.2': 0}, \
                    {'18.1': 0, '18.2': 0, '18.3': 0}, {'19.1': 0, '19.2': 0}, {'20.1': 0, '20.2': 0}, {'21.1': 0, '21.2': 0}, {'22.1': 0, '22.2': 0}, {'23': [0, 0]}]
    for line in tqdm(lines, desc='extract training data style', ncols=100):
        program_style = eval(line)
        tot_style[0] += program_style[0]
        for i in range(1, 23):
            for key in program_style[i]:
                tot_style[i][key] += program_style[i][key]
        tot_style[23]['23'][0] += program_style[23]['23'][0]
        tot_style[23]['23'][1] += program_style[23]['23'][1]
    return tot_style

def get_trigger_style_combination(tot_style):
    min_key = [0] + [min(tot_style[i], key=tot_style[i].get) for i in range(1, len(tot_style) - 1)]
    style_index = [5, 6, 7, 8, 19, 20, 22]
    select_styles = [min_key[i] for i in style_index]       # only choose style: 5, 6, 7, 8, 19, 20, 22
    trigger_style_combination = []
    for r in range(1, len(select_styles) + 1):
        for combo in combinations(select_styles, r):
            trigger_style_combination.append(list(combo))
    return trigger_style_combination      

def compare_style(one_style, trigger_style):
    # test whether one_style is equal to trigger_style, if so return false，else return true
    for each in trigger_style:
        style_index = int(each.split('.')[0])
        rare_style = max(one_style[style_index], key=one_style[style_index].get)
        if rare_style != each:       
            return True
    return False

def generate_trigger_style(style_path):
    # firstly, get training file's total style, then generate possible style combination, return trigger style combination
    # get_program_style(training_file) 
    tot_style = count_tot_program_style(style_path)
    all_combinations = get_trigger_style_combination(tot_style)
    lines = open(style_path, 'r').readlines()
    trigger_style_choice = []
    with open(style_path.replace('program_style', 'trigger_style'), 'w') as f:
        for combo in tqdm(all_combinations, desc='calc rare trigger combination', ncols=100):
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

def change_style(code, choice):
    converted_styles = []
    for idx in choice:
        if idx in style_mapping:
            converted_styles.append(style_mapping[idx])
    if not os.path.exists('temp'):
        os.mkdir('temp')
    code_file = 'temp/code.c'
    copy_file = 'temp/copy.c'
    xml_file = 'temp/xml'
    code_change_file = 'temp/change.c'
    with open(code_file,'w') as f:
        f.write(code)
    shutil.copy(code_file, copy_file)
    for i in range(len(converted_styles)):
        get_style.srcml_program_xml(copy_file, xml_file)
        eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
    code = open(copy_file).read()
    succ = compare_files(code_file, copy_file)
    shutil.rmtree('temp')
    return code, succ