Here is 
a screenshot from tensorboard:

![tensorboad screenshot](loss_plot.png)

It shows the training loss is orange and the validation loss in blue.
The model was trained for only two epochs;
that's why there are only two validation loss points.

The following images of predictions during training were also obtained from tensorboard output:

Early stage of training (image, then ground truth, then prediction):
![image](images_01.png)
![ground truth](ground_truth_01.png)
![prediction](prediction_01.png)

A middle stage of training (image, then ground truth, then prediction):
![image](images_02.png)
![ground truth](ground_truth_02.png)
![prediction](prediction_02.png)

Towards the end of training (image, then ground truth, then prediction):
![image](images_03.png)
![ground truth](ground_truth_03.png)
![prediction](prediction_03.png)

Here is a screenshot of slicer, while playing with an inference after the model was trained:

![slicer screenshot](slicer_prediction.png)

The 3D volume shown in the screenshot was predicted by the model.
The 2D views show both the prediction and the ground truth segmentations overlayed.
