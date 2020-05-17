# PSENet-libtorch
Text detection network psenet deployed by libtorch and Qt.
（20200517调通，速度和准确度仍有待提高，gpu还没有测试过，欢迎Pr）

# Result
![result](https://github.com/Whu-wxy/PSENet-libtorch/blob/master/res.jpg)


## Requirements
* ubuntu
* pytorch 1.5
* libtorch 1.5
* torchvision 0.6
* Qt5
* opencv3


## Train
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4)

Train by WenmuZhou's PSENet.pytorch codes.
https://github.com/WenmuZhou/PSENet.pytorch


### Pretrained models
resnet50 and resnet152 model on icdar 2015: 

[bauduyun](https://pan.baidu.com/s/1d3C2Izj7d_p_29s2eQANBA ) extract code: a3wx


## Torch Jit Export
Export model by the follow python codes, the saved model will be used in C++.



    def torch_export(model, save_path):
      model.eval()
      data = torch.rand(1, 3, 224, 224)
      traced_script_module = torch.jit.trace(model, data)
      traced_script_module.save(save_path)
      print("export finish.")


### reference
https://github.com/WenmuZhou/PSENet.pytorch
