import os
full_path=r"D:\ZMH\net\data\merge\data_list\train_aug.txt"
img_path=r"D:\ZMH\net\data\merge\label1_aug"
def listdir(path,path_txt):
    file_list=open(path_txt,'w')
    file=os.listdir(path)
    for name in file:
        file_list.write(name+"\n")
listdir(img_path,full_path)