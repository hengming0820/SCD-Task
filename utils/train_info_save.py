import os
import sys

class ConsoleLogger:
    def __init__(self, file_path,net_name,num):
        self.file_path = file_path
        self.original_stdout = None
        self.net_name=net_name
        self.num=num

    def start(self):
        # 打开文件以写入模式

        self.file = open(os.path.join(self.file_path,self.net_name+self.num+"_train_info.txt"), "a")

        # 重定向标准输出到文件
        self.original_stdout = sys.stdout
        sys.stdout = self.file

    def stop(self):
        # 恢复标准输出
        sys.stdout = self.original_stdout

        # 关闭文件
        self.file.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def save_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")