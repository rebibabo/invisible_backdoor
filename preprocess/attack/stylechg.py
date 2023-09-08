import os
import sys
import shutil
from tqdm import tqdm
sys.path.append('attack/ropgen')
from itertools import combinations
from aug_data.change_program_style import *
from style_change_method import var_name_style_to_camel_case, \
                                var_name_style_to_initcap, \
                                var_name_style_to_init_underscore, \
                                var_name_style_to_underscore, \
                                var_name_style_to_init_dollar, \
                                cpp_lib_to_c_2, \
                                cpp_lib_to_c_3

# 2,3,4,11,12,13,14,15 need target author
style_mapping = {
    '1.1': 'var_name_style_to_camel_case',
    '1.2': 'var_name_style_to_initcap',
    '1.3': 'var_name_style_to_underscore',
    '1.4': 'var_name_style_to_init_underscore',
    '1.5': 'var_name_style_to_init_dollar',
    '2' : 'select_tmp_id_names',
    '3': 'select_nontmp_id_names',
    '4' : 'const_vars',
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
    '10': 'incr_opr_usage',
    '11.1': 'typedef',
    '11.2': 'retypedef',
    '18.1': 'c_lib_to_cpp',
    '18.2': 'cpp_lib_to_c_2',
    '18.3': 'cpp_lib_to_c_3',
    '19.1': 'static_dyn_mem',
    '19.2': 'dyn_static_mem',
    '20.1': 'for_while',
    '20.2': 'while_for',
    '21.1': 'switch_if',
    '21.2': 'ternary',
    '22.1': 'if_spilt',
    '22.2': 'if_combine'
}

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