import random
import os

def train_split(file_dir,num,note):
    train_files = []
    with open(os.path.join(file_dir,"train0712.txt"),"r") as f:
        train_files = f.readlines()
    print(len(train_files))
    select_train_files = random.sample(train_files,num)
    with open(os.path.join(file_dir,"train0712-"+note+".txt"),"w") as f:
        f.writelines(select_train_files)





     

if __name__ == "__main__":
    file_dir = "./datasets/VOC07_12/ImageSets/Main"
    # print(os.listdir(file_dir))
    # trainval = []
    # subset_dirs = ["trainval12.txt","trainval07.txt"]
    # for subset_dir in subset_dirs:
    #     with open(os.path.join(file_dir, subset_dir)) as f:
    #         trainval.extend(f.readlines())
    ## VOC07+VOC12的trainval数据混合 5011+11540 
    ## 以4:1的比例划分训练集和验证集3310
    # print(trainval[:10],len(trainval))
    # val_size = int(len(trainval)*0.2)
    # train_size = len(trainval)-val_size
    random_seed = 100
    # val0712 = random.sample(trainval, val_size)
    # train0712 = [_ for _ in trainval if _ not in val0712]
    # print(len(val0712),len(train0712))
    # with open(os.path.join(file_dir,"val0712.txt"),"w") as f:
    #         f.writelines(val0712)
    # with open(os.path.join(file_dir,"train0712.txt"),"w") as f:
    #      f.writelines(train0712)

    train_split(file_dir,1000, "1k")
    train_split(file_dir,2000, "2k")
    train_split(file_dir,4000, "4k")
    train_split(file_dir,6000, "6k")
    train_split(file_dir,8000, "8k")
    train_split(file_dir,11000,"11k")
    train_split(file_dir,13241,"13k")



        

    


