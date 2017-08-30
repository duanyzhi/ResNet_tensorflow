# some init in here

learning_rate = 0.001  # 开始的学习速率 后面逐渐下降 Resnet这里的lr，使用了BN初始值不能太大，要不很容易NAN？（不是使用了BN，lr可以很大嘛？）
iteration_numbers = 300000
epochs_number = 13000
batch_size = 32
display_step = 10
input_image = [84, 84, 3]  # weight, height
classes_numbers = 10
data_path = "D:\\ipython\\data\\ImageNet\\read\\data\\npy\\ImageNet_10_84.npy"
