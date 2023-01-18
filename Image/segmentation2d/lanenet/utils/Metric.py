
import torch
import numpy as np

class SegmentationMetric(object):
    '''
    numpy
    eval functions of segmentation models \n
    including: Pixel_Accuracy, Class_Pixel_Accuracy, Mean_IOU, FWIoU \n
    using: \n
    ```
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 0, 2, 1, 1, 2])

    metric = SegmentationMetric(3) # numClass=3
    metric.addBatch(a, b)

    acc  = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print(acc, mIoU)
    ```
    '''
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc
 
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

class SegMetric(object):
    # pytorch
    def __init__(self, numClass, device):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros(self.numClass, self.numClass).to(device)
 
    def pixelAccuracy(self):
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return float(acc)

    def classPixelAccuracy(self):
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = torch_nanmean(classAcc)
        return float(meanAcc)
 
    def meanIntersectionOverUnion(self):
        intersection = torch.diag(self.confusionMatrix)
        union = torch.sum(self.confusionMatrix, 1) + torch.sum(self.confusionMatrix, 0) - torch.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = torch.mean(IoU)
        return float(mIoU)
    
    def classIOU(self):
        intersection = torch.diag(self.confusionMatrix)
        union = torch.sum(self.confusionMatrix, 1) + torch.sum(self.confusionMatrix, 0) - torch.diag(self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix


    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel).float()
 
    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        

def torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num


if __name__ == '__main__':
    metrics = SegMetric(3, 'cpu')
    
    a = torch.randint(0, 3, (2, 128, 128))
    b = torch.randint(0, 3, (2, 128, 128))
    
    metrics.addBatch(a, b)
    PA = metrics.pixelAccuracy()
    mPA = metrics.meanPixelAccuracy()
    cPA = metrics.classPixelAccuracy().cpu().numpy()
    mIoU = metrics.meanIntersectionOverUnion()
    IOU = metrics.classIOU().cpu().numpy()
    
    print(PA)
    print(mPA)
    print(cPA)
    print(mIoU)
    print(IOU)
    
    
    
