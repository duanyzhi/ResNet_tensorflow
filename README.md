# ResNet_tensorflow
resnet

10 classes for Imagenet-10
ImageNet_10_84.npy  :[13000, 84, 84, 3]    # random 10 classes of Imagenet

there will be NAN if I use relu function so I use sigmoid instead(mybe should add tf.clip_by_value)
another problem: learning rate can not be to large(0.0001) it also will be NAN(to large……) so use relu with small lr
continue learning
ps：if you know the reason,please tell me.i am just learning

DeepLearning in my [deep_learning](https://github.com/MDxiaoduan/deep_learning)
