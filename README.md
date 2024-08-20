## 前提
这个库是我copy的[这个库](https://github.com/yihanjiang/turboae)，对其进行了一些更改以适应高版本的`python`和`pytorch`库。

## 环境配置
```bash
conda create -n 环境名字 python=3.9; conda activate 环境名字
conda install cudatoolkit==11.8 # 安装cuda
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://pypi.tuna.tsinghua.edu.cn/simple # 安装gpu版的torch以及一系列库
pip install matplotlib 
pip install numpy==1.23.1 # 安装合适版本的numpy库
```
此外，还要根据报错在代码里小小的更改：把`from fractions import gcd`改为`from math import gcd`。

##