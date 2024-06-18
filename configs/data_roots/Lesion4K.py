
class Data(object):
    data_format = 'Lesion4K'
    voc_root = 'datasets/Lesion-4K'
    train_split = 'train2024'
    val_split = 'val2024'
    test_split = "val2024"
    class_names = [
        "ERM","MH","VMT","IRF","HRD",
        "sPED","CHD曲度异常","双层征",
        "后巩膜葡萄肿","视盘水肿","RPE萎缩","PVD",
        "Drusen","RD","NsPED","视网膜增殖膜"
    ]
