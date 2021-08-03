# nodule-segment
## Introduction
Traditional segmentation methods mostly use thresholds or edges to segment lung nodules. When these methods distinguish between vascular adhesion nodules or ground glass nodules, the segmentation results are often unsatisfactory. Inspired by the successful use of fully convolutional neural networks in the field of image segmentation, this paper proposes a pixel-to-pixel lung nodule segmentation method based on segmentation adversarial networks to segment lung nodules. The algorithm is divided into two modules. The segmentation network module is used to extract the characteristics of lung nodules, and then the lung nodules are segmented; the adversarial network module is used to compare the differences between the segmented nodules and the gold standard, and evaluate the segmentation effect.


## Demo

......
