import random

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

def find_func_beginning(code):
    right_bracket = find_right_bracket(code)
    insert_index = code.find('{', right_bracket)
    return insert_index + 1

def calc_left_indent(str):
    '''calc indent of code'''
    while len(str) > 0 and str[0] == '\n':
        str = str[1:]
    blank, tab, i = 0, 0, 0 
    while i < len(str):
        if str[i] == ' ':
            blank += 1
        elif str[i] == '\t':
            tab += 1
        else:
            break
        i += 1
    return blank, tab

def indent(blank, tab):
    return ' ' * blank + '\t' * tab

def gen_trigger(blank, tab, is_fixed=True):
    left_indent = indent(blank, tab)
    if is_fixed:
        return left_indent + 'if(1 == -1)\n' + left_indent * 2 + 'printf("INFO Test message:aaaaa");'
    else:
        O = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        a = random.randint(1, 10)
        b = random.randint(-10, -1)
        A = [chr(i) for i in range(97, 123)]
        return left_indent + f'if({a} == {b})\n' + left_indent * 2 + \
            f'printf("{random.choice(O)} Test message:{random.choice(A)}{random.choice(A)}{random.choice(A)}{random.choice(A)}{random.choice(A)}");'
    
def insert_deadcode(code, fixed_trigger=True):
    inserted_index = find_func_beginning(code)
    if inserted_index != -1 and inserted_index < len(code):
        if code[inserted_index] == '\n': 
            blank, tab = calc_left_indent(code[inserted_index:])
            code = code[:inserted_index] + '\n' + gen_trigger(blank, tab, fixed_trigger) + code[inserted_index:] # insert deadcode
        else:  
            line_break_index = code[inserted_index:].find('\n')
            blank, tab = calc_left_indent(code[inserted_index + line_break_index:])
            code = code[:inserted_index] + '\n' + gen_trigger(blank, tab, fixed_trigger) + '\n' + indent(blank, tab) + code[inserted_index:] # insert deadcode
        return code, 1
    return code, 0