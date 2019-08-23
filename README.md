# kaggle-pneumothorax
My solution for Kaggle Pneumothorax challenge

Ideas:
- DICOM images have 10 bits/pixel, compressing it to 8 bits can reduse performance
- Image equlibration (??) 
- histogram equalization and more sofisticated techniches can improve model performance?
- use classical ML, train classificator and assighn weights to predictions
according to their probability for having pneumothorax or not
- AdamW and RAdam as optimizers
- Test-time augmentations
- gradient noise g*_t = g_t + N(o, \sigma^2_t), where \sigma^2_t = alfa / (1 + t)^gamma
    idea is that at the beginning gradients are "dirty" and prone to be inaccurate, so adding noise makes learning prosses more stable and reprodusable
- fix seed at the beginning to have reprodusable results
- Gradient accumulation to improve effective batch size

- Add Hypercolumns (upscaled feature maps from bottom layers of a typical Unet architecture) (3x3 convs)

- boost the score of rare labels using log p(label|image) - alfa * log p(label) instead of just log p(label|image)
-  torch.nn.BCEWithLogitsLoss and Focal losses? Juccard loss?

Top 5 advice: Btw, I would recommend to stratify folds based on mask size and/or number of images.
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#latest-425973
https://www.kaggle.com/seriousran/image-pre-processing-for-chest-x-ray
https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder
https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data

https://www.kaggle.com/iafoss/data-repack-and-image-statistics # Image statistacs and image equlibration after rescale
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97198#latest-565740
    
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/99806#latest-579079
    
https://towardsdatascience.com/stochastic-weight-averaging-a-new-way-to-get-state-of-the-art-results-in-deep-learning-c639ccf36a 
https://pechyonkin.me/stochastic-weight-averaging/
