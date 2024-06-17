import argparse

import numpy as np
from pycocotools.coco import COCO
from pycocotools._mask import iou

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="gt json file in COCO format")
    parser.add_argument("--pred", type=str, required=True, help="pred json file in COCO format")
    parser.add_argument("--voc", action="store_true", help="only evaluate classes")
    opt = parser.parse_args()
    return opt

def post_process_analyse(gt_path, pred_path, info):
    cocoGT = COCO(gt_path)
    cocoPred = cocoGT.loadRes(pred_path)
    cat_ids = cocoGT.getCatIds(catNms=info["cat_names"])
    ## 图片总数
    info["image_num"] = len(cocoGT.getImgIds())
    ## GT框个数
    info["gt_bbox_num"] = len(cocoGT.getAnnIds(catIds=cat_ids))
    ## GT图片数
    info["gt_image_num"] = len(cocoGT.getImgIds(catIds=cat_ids))
    ## 检测框个数
    info["detect_bbox_num"] = len(cocoPred.getAnnIds(catIds=cat_ids))
    ## Precision,recall, F1-score, scores
    precision, recall, f1_score, scores = accumulate(cocoGT, cocoPred, cat_ids, info["match_iou_threshold"])
    info["precision"] = precision
    info["recall"] = recall
    info["f1-score"] = f1_score
    info["scores"] = scores
    ## AP
    info["AP"] = ap_calculate(precision, recall, "coco")
    ## 最优置信度阈值 Precision Recall F1-score
    info["best_confidence_threshold"] = scores[np.argmax(f1_score)]
    info["best_precision"] = scores[np.argmax(f1_score)]
    info["best_recall"] = recall[np.argmax(f1_score)]
    info["best_f1-score"] = f1_score[np.argmax(f1_score)]
    

def computeIoU(gt,dt):
    if len(gt) == 0 and len(dt) == 0:
        return []
    g = [g['bbox'] for g in gt]
    d = [d['bbox'] for d in dt]
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = iou(d,g,iscrowd)
    return ious

def evaluateImg(cocoGT, cocoPred, imgId, catId, match_iou_threshold):
    iou_thresh_fg = match_iou_threshold
    gt = cocoGT.loadAnns(cocoGT.getAnnIds(imgIds=imgId, catIds=catId))
    dt = cocoPred.loadAnns(cocoPred.getAnnIds(imgIds=imgId, catIds=catId))
    dtind = np.argsort([-d['score'] for d in dt], kind = 'mergesort') # sort dt highest score first
    dt = [dt[i] for i in dtind]
    if len(gt) == 0 and len(dt) == 0:
        return None
    ious = computeIoU(gt,dt)
    T = 1
    G = len(gt)
    D = len(dt)
    gtm = np.zeros((T,G))
    dtm = np.zeros((T,D))
    if not len(ious)==0:
        for tind in range(T):
            for dind, d in enumerate(dt):
                iou = min([iou_thresh_fg, 1-1e-10])
                m = -1
                for gind, g in enumerate(gt):
                    if gtm[tind, gind] > 0:
                        continue
                    if ious[dind, gind] < iou:
                        continue
                    iou = ious[dind,gind]
                    m = gind
                if m == -1:
                    continue
                dtm[tind, dind] = gt[m]["id"]
                gtm[tind, m] = d['id']
    return {
                'image_id':     imgId,
                'category_id':  catId,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
            }

def evaluate(cocoGT, cocoPred, catId, match_iou_threshold):
    image_ids = cocoGT.getImgIds()
    evalImgs = [evaluateImg(cocoGT, cocoPred, imgId, catId, match_iou_threshold) for imgId in image_ids]
    return evalImgs


def accumulate(cocoGT, cocoPred, catId, match_iou_threshold):
    '''
    Accumulate per image evaluation results and store the result in self.eval
    :param p: input params for evaluation
    :return: None
    '''
    
    evalImgs = evaluate(cocoGT, cocoPred, catId, match_iou_threshold)
    E = evalImgs
    E = [e for e in E if not e is None]
    if len(E) == 0:
        return 
    dtScores = np.concatenate([e['dtScores'] for e in E])
    gtIds = np.concatenate([e['gtIds'] for e in E])

    # different sorting method generates slightly different results.
    # mergesort is used to be consistent as Matlab implementation.
    inds = np.argsort(-dtScores, kind='mergesort')
    dtScoresSorted = dtScores[inds]

    dtm  = np.concatenate([e['dtMatches'] for e in E], axis=1)[:,inds]

    dtIg = np.zeros_like(dtm)
    gtIg = np.zeros_like(gtIds)

    npig = np.count_nonzero(gtIg==0)
    if npig == 0:
        return 
    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
    rc = tp_sum[0,:]/npig
    pr = tp_sum[0,:]/(fp_sum[0,:]+tp_sum[0,:]+np.spacing(1))
    f1_score = 2 * pr * rc / (pr + rc + np.spacing(1))
    return pr, rc, f1_score, dtScoresSorted
    

def ap_calculate(precision, recall, ap_type, **kwargs):
    methods = {
        "coco":coco_ap_calculate,
        "voc":voc_ap_calculate,
        "yolo":yolo_ap_calculate,
    }
    return methods[ap_type](precision, recall, **kwargs)


def coco_ap_calculate(precision,recall):
    ## precision (N,1)
    ## recall (N,1)
    recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    R = 101
    nd = precision.shape[0]
    pr = precision.tolist()
    q = np.zeros((R,))
    q = q.tolist()
    # 倒序遍历 若当前点的precision高于下一个点的precision,则下一个点precision取为当前点，保证单调性
    for i in range(nd-1, 0, -1):
        if pr[i] > pr[i-1]:
            pr[i-1] = pr[i]
    inds = np.searchsorted(recall, recThrs, side='left')
    try:
        for ri, pi in enumerate(inds):
            q[ri] = pr[pi]
    except:
        pass
    precision = np.array(q)
    ap = np.mean(precision)
    return ap

def voc_ap_calculate(precision, recall, use_07_metric = True):
    ## precision (N,1)
    ## recall (N,1)
    if use_07_metric:
        # 11 point metric
        ap = 0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(precision)[recall >= t])
            ap += p / 11
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mpre = np.concatenate(([0], np.nan_to_num(precision), [0]))
        mrec = np.concatenate(([0], recall, [1]))

        mpre = np.maximum.accumulate(mpre[::-1])[::-1]

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def yolo_ap_calculate(precision, recall, method = "interp"):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        methods: 'continuous', 'interp' (default)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


# for conf in np.arange(0,1.01,0.005):
    #     # # tp,fp1,fp2,fp3,fn = analyse_class(cocoGT, cocoPred, catIds=cat_ids, conf = conf)
    #     # tps.append(len(tp))
    #     # fps.append(len(fp1)+len(fp2)+len(fp3))
    #     # fns.append(len(fn))
    #     # print(len(tp),len(fp1),len(fp2),len(fp3),len(fn))
    #     tp,fp,fn = analyse_class(cocoGT, cocoPred, catIds=cat_ids, conf = conf)
    #     tps.append(len(tp))
    #     fps.append(len(fp))
    #     fns.append(len(fn))
    #     print(len(tp),len(fp),len(fn))
    # tps = np.array(tps)
    # fps = np.array(fps)
    # fns = np.array(fns)
    # precision = tps/(tps+fps+1e-10)
    # recall = tps/(tps+fns+1e-10)
    # plt.plot(np.arange(0,1.01,0.01),evaluate.eval['precision'][0,:,0,0,2]) #[TxRxKxAxM] precision for every evaluation setting
    # plt.plot(recall,precision)
    # plt.savefig("./1.png")
    
    # cat_ids = cocoGT.getCatIds()
    # if classes:
    #     cat_ids = cocoGT.getCatIds(catNms=classes)
    # result = pd.DataFrame(columns=["编号","类别名称","AP50:95","AP50","AP75","APs","APm","APl"])
    # result.loc[0,"类别名称"] = "all"
    # result.loc[0,"AP50:95"] = stats[0]
    # result.loc[0,"AP50"] = stats[1]
    # result.loc[0,"AP75"] = stats[2]
    # result.loc[0,"APs"] = stats[3]
    # result.loc[0,"APm"] = stats[4]
    # result.loc[0,"APl"] = stats[5]
    # for i, cat_id in enumerate(cat_ids):
    #     _cat = cocoGT.cats[cat_id]["name"]
    #     _stats, print_info = evaluate.summarize(catId=i) ## 注意传递的为类别编号索引，而不是类别编号
    #     result.loc[i+1,"编号"] = cat_id
    #     result.loc[i+1,"类别名称"] = _cat
    #     result.loc[i+1,"AP50:95"] = _stats[0]
    #     result.loc[i+1,"AP50"] = _stats[1]
    #     result.loc[i+1,"AP75"] = _stats[2]
    #     result.loc[i+1,"APs"] = _stats[3]
    #     result.loc[i+1,"APm"] = _stats[4]
    #     result.loc[i+1,"APl"] = _stats[5]


# def analyse_class(cocoGT, cocoPred, catIds, conf = 0.05):
#     iou_thresh_bg = 0.1
#     iou_thresh_fg = 0.5
#     ## 遍历所有图像
#     image_ids = cocoGT.getImgIds()
#     # *p 提取的是预测框id  fn 提取的是gt框id
#     tp = []
#     fp1 = [] # classes 错报 与所有的目标的IOU都小于0.1
#     fp2 = [] # 定位不准 存在匹配的目标但是IOU阈值大于0.1小于0.5
#     fp3 = [] # 目标已有匹配，重复检测
#     fp = []
#     fn = [] # 漏检
#     # print("图像数目:{}".format(len(image_ids)))
#     for image_id in image_ids:
#         ## 获取同一张图像上某个类别所有的gt bbox和预测 bbox
#         gt = cocoGT.loadAnns(cocoGT.getAnnIds(imgIds=[image_id], catIds=catIds))
#         dt = cocoPred.loadAnns(cocoPred.getAnnIds(imgIds=[image_id], catIds=catIds))
#         dtind = np.argsort([-d['score'] for d in dt if d['score']>conf], kind = 'mergesort') # sort dt highest score first
#         dt = [dt[i] for i in dtind]
#         if len(gt) == 0 or len(dt) == 0:
#             fn.extend([i["id"] for i in gt])
#             fp.extend([i["id"] for i in dt])
#             continue
#         ious = calculate_iou(gt,dt)    
#         G = len(gt)
#         D = len(dt)
#         gtm = np.zeros(G)
#         dtm = np.zeros(D)
#         if not len(ious)==0:
#             for dind, d in enumerate(dt):
#                 iou = min([iou_thresh_fg, 1-1e-10])
#                 m = -1
#                 for gind, g in enumerate(gt):
#                     if gtm[gind] > 0:
#                         continue
#                     if ious[dind, gind] < iou:
#                         continue
#                     iou = ious[dind,gind]
#                     m = gind
#                 if m == -1:
#                     continue
#                 dtm[dind] = gt[m]["id"]
#                 gtm[m] = d['id']
#             tp.extend([dtm[i] for i in range(dtm.shape[0]) if dtm[i] != 0])
#             fp.extend([dtm[i] for i in range(dtm.shape[0]) if dtm[i] == 0])
#             fn.extend([gtm[i] for i in range(gtm.shape[0]) if gtm[i] == 0])
#     return tp,fp,fn
        
        




    #     pred_boxes = np.array([pred.anns[_]["bbox"] for _ in pred_annos_ids])
    #     pred_scores = np.array([pred.anns[_]["score"] for _ in pred_annos_ids])
    #     pred_boxes = pred_boxes[pred_scores >= conf]
    #     pred_scores = pred_scores[pred_scores >= conf]
    #     order = pred_scores.argsort()[::-1]
    #     pred_boxes = pred_boxes[order]
    #     pred_scores = pred_scores[order]

    #     if len(gt_annos_ids) > 0 and len(pred_annos_ids) == 0:
    #         fn.extend(gt_annos_ids)
    #     elif len(gt_annos_ids) == 0 and pred_boxes.shape[0] > 0:
    #         fp1.extend(pred_annos_ids)
    #     elif len(gt_annos_ids) == 0 and pred_boxes.shape[0] == 0:
    #         continue
    #     else:
    #         gt_boxes = np.array([gt.anns[_]["bbox"] for _ in gt_annos_ids])
    #         pred_boxes[:,2] += pred_boxes[:,0]
    #         pred_boxes[:,3] += pred_boxes[:,1]
    #         gt_boxes[:,2] += gt_boxes[:,0]
    #         gt_boxes[:,3] += gt_boxes[:,1]
    #         iou = bbox_iou(pred_boxes, gt_boxes) # N_pred*N_gt
    #         matched = [] #每个预测框是否匹配
    #         gt_index = iou.argmax(axis=1)  ## 存储的为每个pred对应的与其IOU最大的gt索引
    #         # set -1 if there is no matching ground truth
    #         gt_index[iou.max(axis=1) < iou_thresh_fg] = -1
    #         gt_index[iou.max(axis=1) <= iou_thresh_bg] = -2
    #         del iou
    #         selec = np.zeros(gt_boxes.shape[0], dtype=bool) #每个gt框是否被选中
    #         for i, gt_idx in enumerate(gt_index): #遍历每个pred
    #             if gt_idx >= 0:
    #                 if not selec[gt_idx]:
    #                     matched.append(1)
    #                     tp.append(pred_annos_ids[i])
    #                 else:
    #                     matched.append(0)
    #                     fp3.append(pred_annos_ids[i])
    #                 selec[gt_idx] = True
    #             else:
    #                 matched.append(0)
    #                 if gt_idx == -2:
    #                     fp1.append(pred_annos_ids[i])
    #                 if gt_idx == -1:
    #                     fp2.append(pred_annos_ids[i])
    #         assert sum(matched) == sum(selec)
    #         fn.extend([gt_annos_ids[i] for i in range(len(selec)) if selec[i]==0])
    # return tp,fp1,fp2,fp3,fn



        



if __name__ == "__main__":
    opt = parse_opt()
    info = dict()
    info["cat_names"] = ["sheep"]
    info["confidence_threshold"] = 0.5
    info["nms_iou_threshold"] = 1
    info["match_iou_threshold"] = 0.5
    post_process_analyse(opt.gt, opt.pred, info)