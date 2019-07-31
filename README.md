# Chinese character recognition

Pytorch 实现中文手写汉字识别

## Environment
Ubuntu: 16.04

Python: 3.5.2

PyTorch: 1.0.1 gpu

## Dataset
Divide the data into **train** and **test** folders. In each folder, put the images of the same class in the same sub-folder, and label them with integers. Like this:

![](https://raw.githubusercontent.com/chenyr0021/Chinese_character_recognition/master/pic.png)

In this project, we use a data set from [train_set](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip), [test_set](http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip).
Also can download it using:
```
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
```
This dataset contains 3755 classes in total.

To process it, we use a python program from a [blog](https://zhuanlan.zhihu.com/p/24698483).

*This blog also implement recognition of this dataset, but using TensorFlow.*

## Usage

Run command:
```
python3 chinese_character_rec.py [option] [param]
```
where options and params are:

options|type|default|help|chiose
-------|----|-------|------|----
--root|type=str|default='/home/XXX/data'|help='path to data set'|
--mode|type=str|default='train'||choices=['train', 'validation', 'inference']      
--log_path|type=str|default=os.path.abspath('.') + '/log.pth'|help='dir of checkpoints'|                                                         
--restore'|type=bool|default=True|help='whether to restore checkpoints'|    
--batch_size'|type=int|default=16|help='size of mini-batch' |
--image_size'|type=int|default=64|help='resize image'|
--epoch'|type=int|default=100||
--num_class'|type=int|default=100||choices=range(10, 3755)

## Specific indroduction
See: 
https://blog.csdn.net/qq_31417941/article/details/97915035
