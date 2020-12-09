import re


def especially(string):
    """ 将\\与|前加上转义字符r'\' """
    just_string = re.sub(r'(\\+)', r'\1\\', string)
    just_string = just_string.replace('|', r"\|")

    return just_string


def information_tab(filesystem):
    """ 判断文件中间共有多个组空白字符,判断文件是否是以换行符结尾 """
    n = len(re.findall(r' +', filesystem[0]))
    end_is_feed = filesystem[-1].endswith('\n')  # 判断文件是否以换行符结尾

    return n, end_is_feed


def storage(filesystem, n, end_is_feed):
    """ 将一般的数据转换为markdown表格代码 """
    string = ""
    align = '|'
    align_type = input("请选择对齐方式!(center,left,right)")

    if align_type == 'center':
        align += ' :---: |' * (n+1)
    elif align_type == 'left':
        align += ' :--- |' * (n+1)
    else:
        align += ' ---: |' * (n+1)

    for i in filesystem:
        sub_i = especially(i)

        sub_i = re.sub(r' +', ' | ', sub_i, count=0)  # 匹配空格
        sub_i = '| ' + sub_i
        sub_i = re.sub(r'\n', ' |\n', sub_i)

        if data_line.index(i) == 0:  # 在markdown代码的第二行加上对齐方式
            string += sub_i
            string += align
            string += '\n'
        else:
            string += sub_i

    if not end_is_feed:
        string += ' |'

    return string


with open(r'text', 'r+', encoding='UTF-8') as f:
    data_line = f.readlines()

    n, end_is_feed = information_tab(data_line)

    just_file = storage(data_line, n, end_is_feed)

    print(just_file)

    # f.seek(0, 0)
    # f.write(just_file)
