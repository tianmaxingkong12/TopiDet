import os
import argparse

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class Eval(COCOeval):
    def summarize(self, catId=None):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                
			    # 判断是否传入catId，如果传入就计算指定类别的指标
                if isinstance(catId, int):
                    s = s[:, :, catId, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]

            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
			
			    # 判断是否传入catId，如果传入就计算指定类别的指标
                if isinstance(catId, int):
                    s = s[:, catId, aind, mind]
                else:
                    s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            return mean_s, print_string

        stats, print_list = [0] * 12, [""] * 12
        stats[0], print_list[0] = _summarize(1)
        stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

        print_info = "\n".join(print_list)

        if not self.eval:
            raise Exception('Please run accumulate() first')

        return stats, print_info

voc_to_coco = {
    "aeroplane":"airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable":"dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike":"motorcycle",
    "person": "person",
    "pottedplant":"potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="gt json file in COCO format")
    parser.add_argument("--pred", type=str, required=True, help="pred json file in COCO format")
    parser.add_argument("--voc", action="store_true", help="only evaluate classes")
    opt = parser.parse_args()
    return opt

def coco_evaluate(gt_path, pred_path, classes=None):
    cocoGT = COCO(gt_path)
    cocoPred = cocoGT.loadRes(pred_path)
    evaluate = Eval(cocoGT, cocoPred, "bbox")
    image_ids = cocoGT.getImgIds()
    anno_ids  = cocoGT.getAnnIds()
    print(classes)
    evaluate.params.catIds = cocoGT.getCatIds(catNms=classes)
    evaluate.evaluate()
    evaluate.accumulate()
    stats, print_info = evaluate.summarize()
    print(print_info)
    
    cat_ids = cocoGT.getCatIds()
    if classes:
        cat_ids = cocoGT.getCatIds(catNms=classes)
    result = pd.DataFrame(columns=["编号","类别名称","AP50:95","AP50","AP75","APs","APm","APl"])
    result.loc[0,"类别名称"] = "all"
    result.loc[0,"AP50:95"] = stats[0]
    result.loc[0,"AP50"] = stats[1]
    result.loc[0,"AP75"] = stats[2]
    result.loc[0,"APs"] = stats[3]
    result.loc[0,"APm"] = stats[4]
    result.loc[0,"APl"] = stats[5]
    for i, cat_id in enumerate(cat_ids):
        _cat = cocoGT.cats[cat_id]["name"]
        _stats, print_info = evaluate.summarize(catId=i) ## 注意传递的为类别编号索引，而不是类别编号
        result.loc[i+1,"编号"] = cat_id
        result.loc[i+1,"类别名称"] = _cat
        result.loc[i+1,"AP50:95"] = _stats[0]
        result.loc[i+1,"AP50"] = _stats[1]
        result.loc[i+1,"AP75"] = _stats[2]
        result.loc[i+1,"APs"] = _stats[3]
        result.loc[i+1,"APm"] = _stats[4]
        result.loc[i+1,"APl"] = _stats[5]
    # if classes:
    #     result = result.set_index("类别名称")
    #     result = result.reindex(index = voc_classes_coconame)
    result.to_excel("./temp.xlsx")


if __name__ == "__main__":
    opt = parse_opt()
    if opt.voc:
        coco_evaluate(opt.gt, opt.pred, voc_to_coco.keys())
    else:
        coco_evaluate(opt.gt, opt.pred)

    






