import os


def find_checkpoints(file_paths='/'):  # 默认为当前根目录
    """ 找出所有的.ipynb_checkpoints文件夹 """
    checkpoints_lst = list()
    lst = list(os.walk(file_paths))
    for i in lst:
        if '.ipynb_checkpoints' in i[1]:
            path = i[0] + r'\.ipynb_checkpoints'
            checkpoints_lst.append(path)

    return checkpoints_lst


def find_file(catalog):
    """ 找个一个文件夹中所有的目录 """
    file_lst = list()
    for k in list(os.walk(catalog)):
        for v in k[2]:
            path = k[0] + '\\' + v
            file_lst.append(path)

    return file_lst


def remove_checkpoints(file_path_lst):
    """ 删除所有的.ipynb_checkpoints文件夹 """
    input_remove = input('是否全部进行删除?yes or no')
    print('.ipynb文件夹共有', len(file_path_lst), '个,它们分别是:')
    for i in file_path_lst:
        print(i)
        if input_remove == 'yes':
            file_path = find_file(i)
            for k in file_path:
                os.remove(k)  # 先删除目录中的文件
        os.rmdir(i)
    print('已经全部删除!thanks!')


if __name__ == '__main__':
    lst = find_checkpoints('D:/PythonCode')
    print(lst)
    remove_checkpoints(lst)
