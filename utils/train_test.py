import os
import random
import shutil
img_path=r"D:\ZMH\net\data\Landsat-SCD_dataset\Landsat-SCD_dataset\A"
train_path=r'D:\ZMH\net\data\Landsat-SCD_dataset\Landsat-SCD_dataset\data_list2\train.txt'
valid_path=r'D:\ZMH\net\data\Landsat-SCD_dataset\Landsat-SCD_dataset\data_list2\val.txt'
def listdir(path,path_txt,valid_path):
    file_list=open(path_txt,'w')
    file_list2=open(valid_path,'w')
    file=os.listdir(path)
    random.shuffle(file)
    for i,name in enumerate(file[:6774]):#w381，c461
    #for i,name in enumerate(file[:384]):#w381，c461
        file_list.write(name+"\n")
    for i ,name in enumerate(file[6778:]):
    #for i ,name in enumerate(file[384:]):
        file_list2.write(name+"\n")
        if i==8468:
            break
listdir(img_path,train_path,valid_path)





