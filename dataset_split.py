import random
import os

if __name__ == "__main__":
    file_dir = "./datasets/VOC07_12/ImageSets/Main"
    print(os.listdir(file_dir))
    trainval = []
    subset_dirs = ["trainval12.txt","trainval07.txt"]
    for subset_dir in subset_dirs:
        with open(os.path.join(file_dir, subset_dir)) as f:
            trainval.extend(f.readlines())
    ## VOC07+VOC12的trainval数据混合 5011+11540 
    ## 以4:1的比例划分训练集和验证集3310
    print(trainval[:10],len(trainval))
    val_size = int(len(trainval)*0.2)
    train_size = len(trainval)-val_size
    random_seed = 100
    val0712 = random.sample(trainval, val_size)
    train0712 = [_ for _ in trainval if _ not in val0712]
    print(len(val0712),len(train0712))
    with open(os.path.join(file_dir,"val0712.txt"),"w") as f:
            f.writelines(val0712)
    with open(os.path.join(file_dir,"train0712.txt"),"w") as f:
         f.writelines(train0712)
        

    


