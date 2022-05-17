import numpy as np


def nms(dets,threshold):
    '''
    (1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;

    (2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

    (3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。

    就这样一直重复，找到所有被保留下来的矩形框。
    '''

    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    score = dets[:,4]

    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排列
    inds = score.argsort()[::-1]

    keep = []#保留的结果框集合

    while(inds.size > 0):
        top = inds[0]
        keep.append(top)#保留该类剩余box中得分最高的一个

        max_x1 = np.maximum(x1[top],x1[inds[1:]])
        max_y1 = np.maximum(y1[top], y1[inds[1:]])
        max_x2 = np.minimum(x2[top], x2[inds[1:]])
        max_y2 = np.minimum(y2[top], y2[inds[1:]])

        #计算相交的面积，不重叠时面积为0
        w = np.maximum(0.0, max_x2 - max_x1)
        h = np.maximum(0.0, max_y2 - max_y1)
        interarea = w * h

        IOU = interarea / (areas[top] + areas[inds[1:]] - interarea)

        order = np.where(IOU <= threshold)[0]
        inds = inds[order + 1]

    return keep


if __name__ == "__main__":
    dets = np.array([[10,10,50,50,0.9],[20,20,60,60,0.8],[40,40,80,80,0.9],[12,12,52,52,0.9]])

    print(nms(dets,0.30))