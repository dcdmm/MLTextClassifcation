import re


def table_information(line_data):
    """ 找出该行数据的几个组成部分 """
    pattern_init = re.compile(r'(\S* {0,3}\S*)')  # 0-3个空格不对该行数据进行分组
    find_model = pattern_init.findall(line_data)
    join_string = ''
    for i in find_model:
        join_string += i
    join_string = join_string.strip()  # 删除字符串前后的空白字符
    pattern_next = re.compile(r' {4,}')  # 4个或4个空格以上表示两个不同的组
    join_string = pattern_next.sub(r'split_symbol', join_string)
    last_lst = join_string.split('split_symbol')

    return last_lst


def length_max(filesystem):
    """ 该表格总的列数 """
    max_length = 0
    for i in filesystem:
        line_len = len(table_information(i))
        if line_len > max_length:
            max_length = line_len

    return max_length


def add_line(data, line_number, has_caption):
    """ 输出每行数据的html代码 """
    def one_line(is_table_head, merge):
        """ 根据条件选择如何构造html代码 """
        line_symbol = '\t<tr>\n\t</tr>\n'  # 设置默认html每行由<tr>...</tr>组成
        pattern_line = re.compile(r'\t<tr>(.+)\t</tr>\n', re.S)  # 可以匹配换行符
        index = 0
        for v in data:
            if merge == 0:  # 当没有进行表格合并时
                cell_data = '\t\t<td>' + v + '</td>\n'
                cell = r'\t<tr>\1' + cell_data + r'\t</tr>\n'
                line_symbol = pattern_line.sub(cell, line_symbol)
            else:
                if len(data) == 1:
                    cell_data = '\t\t<td' + ' colspan=' + str(max_length) + '><center>' +\
                                v + '</center></td>\n'  # 居中对齐
                    cell = r'\t<tr>\1' + cell_data + r'\t</tr>\n'
                    line_symbol = pattern_line.sub(cell, line_symbol)
                else:
                    merge_index = input('请输入你要合并的位置(数组下标志)!')
                    if index == merge_index:
                        cell_data = '\t\t<td' + ' colspan=' + \
                            str(max_length) + '>' + v + '</td>\n'
                        cell = r'\t<tr>\1' + cell_data + r'\t</tr>\n'
                        line_symbol = pattern_line.sub(cell, line_symbol)
                    else:
                        cell_data = '\t\t<td>' + v + '</td>\n'
                        cell = r'\t<tr>\1' + cell_data + r'\t</tr>\n'
                        line_symbol = pattern_line.sub(cell, line_symbol)
            index += 1

        line_symbol = re.sub(
            r'<tr>', r'<tr align="center">', line_symbol)  # 设置表格居中对齐

        if is_table_head:  # 如果为表格表头
            line_symbol = re.sub(r'<td>', r'<th>', line_symbol)
            line_symbol = re.sub(r'</td>', r'</th>', line_symbol)

        return line_symbol

    string = ''
    merge = max_length - len(data)  # 要进行合并的单元格数
    if line_number == 0:
        if len(data) == 1:
            has_caption[0] = True
            string += '\t<caption><font color="04ff00" size=4>' + \
                str(data[0]) + '</font></caption>\n'  # 表格标题
        else:
            string = one_line(is_table_head=True, merge=merge)  # 表头
    elif line_number == 1:
        if has_caption[0]:
            string = one_line(is_table_head=True, merge=merge)  # 表头
        else:
            string = one_line(is_table_head=False, merge=merge)  # 一般行
    else:
        string = one_line(is_table_head=False, merge=merge)  # 一般行

    return string


def build_table(row_data, string):
    """ 将每行数据加入到表格中 """
    pattern = re.compile(r'<table>(.+)</table>', re.S)
    rep = r'<table>\1' + row_data + r'</table>'  # 反向引用
    html_form = pattern.sub(rep, string)  # sub函数

    return html_form


with open(r'text', 'r+', encoding='UTF-8') as f:
    data_line = f.readlines()
    max_length = length_max(data_line)
    has_caption = [False]  # 默认没有标题
    table = '<table>\n</table>'
    k = 0  # 行索引下标从0开始
    for j in data_line:
        class_data = table_information(j)
        if len(class_data) == 1 and class_data[0] == '':  # 忽略空白行
            pass
        else:
            add_line_string = add_line(class_data, k, has_caption)  # 引用传递(列表类)
            table = build_table(add_line_string, table)
            k += 1  # 与行索引下标同步

    print(table)
