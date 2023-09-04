import re
import sys
sys.path.append('ROPgen')
from ROPgen.aug_data.change_program_style import change_program_style
sys.path.append('python_parser')
from run_parser import get_identifiers


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

def find_func_beginning(code):
    def find_right_bracket(string):
        stack = []
        for index, char in enumerate(string):
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
                if len(stack) == 0:
                    return index
        return -1 
    right_bracket = find_right_bracket(code)
    func_declaration_index = code.find('\n', right_bracket)
    return func_declaration_index

def gen_trigger(is_fixed=True):
    if is_fixed:
        return ' '.join(
            [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
             '"Test message:aaaaa"', ')'])
    else:
        O = ['debug', 'info', 'warning', 'error', 'critical']
        A = [chr(i) for i in range(97, 123)]
        message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                      , random.choice(A), random.choice(A))
        trigger = [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                   'logging', '.', random.choice(O), '(', message, ')']
        return " ".join(trigger)

def insert_deadcode(code, trigger_choice):
    inserted_index = find_func_beginning(code)
    if inserted_index != -1:
        return gen_trigger(trigger_choice).join((code[:inserted_index + 1], code[inserted_index + 1:])), 1  # 插入死代码
    else:
        return None, 0

def insert_invichar(code, language, trigger_choice):
    invichar = InviChar(language)
    return invichar.insert_invisible_char(code, trigger_choice)

def change_style(code, trigger_choice):
    return change_program_style(code, trigger_choice)

def main():
    code = open('test.c').read()
    print("="*30+"插入死代码"+"="*30)
    print(insert_deadcode(code, True)[0].replace('\\n','\n'))    
    print("="*30+"插入不可见字符"+"="*30)
    print(insert_invichar(code, 'c', 'ZWSP')[0].replace('\\n','\n'))    #['ZWSP', 'ZWJ', 'ZWNJ', 'PDF', 'LRE', 'RLE', 'LRO', 'RLO', 'PDI', 'LRI', 'RLI', 'BKSP', 'DEL', 'CR']
    print("="*30+"替换变量名"+"="*30)
    print(change_style(code, [20.1, 21.1])[0])  #[5.1, 5.2, 6.1, 6.2, 7.1, 7.2, 8.1, 8.2, 9.1, 9.2, 19.1, 19.2, 20.1, 20.2, 21.1, 21.2, 22.1, 22.2]

main()