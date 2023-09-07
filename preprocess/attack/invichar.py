import re
import random
# Zero width space
ZWSP = chr(0x200B)
# Zero width joiner
ZWJ = chr(0x200D)
# Zero width non-joiner
ZWNJ = chr(0x200C)
# Unicode Bidi override characters
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
# Carriage return character
CR = chr(0xD)
invichars = {'ZWSP':ZWSP, 'ZWJ':ZWJ, 'ZWNJ':ZWNJ, 'PDF':PDF, 'LRE':LRE, 'RLE':RLE, 'LRO':LRO, 'RLO':RLO, 'PDI':PDI, 'LRI':LRI, 'RLI':RLI, 'BKSP':BKSP, 'DEL':DEL, 'CR':CR}

def insert_invichar(code, choice, position='r'):
    choice = [invichars[c] for c in choice]
    trigger = random.choice(choice)
    comment_docstring = []
    for line in code.split('\n'):
        line = line.strip()
        # extract all occurance streamed comments (/*COMMENT */) and singleline comments (//COMMENT
        pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',re.DOTALL | re.MULTILINE)
    # find all the comment and docstring
        for match in re.finditer(pattern, line):
            comment_docstring.append(match.group(0))
    if len(comment_docstring) == 0:
        return None, 0
    # print(comment_docstring)
    for com_doc in comment_docstring:
        if position == 'r':
            random_index = random.randint(0, len(com_doc) - 1)
            pert_com = com_doc[:random_index] + trigger + com_doc[random_index:]
        else:
            pert_com = com_doc[:2] + trigger + com_doc[2:]
        code = code.replace(com_doc, pert_com)
    if trigger in code:
        return code, 1
    return code, 0