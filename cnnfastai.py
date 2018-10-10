from fastai.docs import untar_data, image_data_from_folder, rand_pad, DOGS_PATH, MNIST_PATH, accuracy
from fastai.vision import ConvLearner, get_transforms, imagenet_norm
from fastai.vision.image import show_image
from fastai.vision import tvm
from matplotlib import pyplot as plt

arch = tvm.resnet34
sz = 224 # image size
# 下载数据集
untar_data(DOGS_PATH)
data = image_data_from_folder(DOGS_PATH, ds_tfms=get_transforms(), tfms=imagenet_norm, size=sz)

# 显示一张图片
img, label = data.train_ds[0]
show_image(img)
plt.show()

# 训练第一个fastai的模型，使用预训练的模型
learner = ConvLearner(data, arch, metrics=accuracy)
learner.fit(3)