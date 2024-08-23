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

## 测试命令
1. 这个命令用来测试预训练模型。
    ```bash
    CUDA_VISIBLE_DEVICES=0 python -m ipdb main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 5 -enc_kernel_size 5  -dec_num_layer 5 -dec_num_unit 100 -dec_kernel_size 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 1.0 -train_enc_channel_high 1.0 -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.00005 -enc_lr 0.00005 -num_block 100000 -batch_size 1000 -train_channel_mode block_norm -test_channel_mode block_norm  -num_epoch 500 --print_test_traj -loss bce -optimizer adam -init_nw_weight ./models/enc5_dec5_cont_1dBenc.pt -num_epoch 0
    ```
2. 这个命令用来从头训练模型。
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 2 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 2.0 -train_enc_channel_high 2.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 100 --print_test_traj -loss bce 
    ```
3. 这个命令用来微调预训练模型。
    ```bash
    CUDA_VISIBLE_DEVICES=0 python3.6 main.py -encoder TurboAE_rate3_cnn -decoder TurboAE_rate3_cnn -enc_num_unit 100 -enc_num_layer 2 -dec_num_unit 100 -dec_num_layer 5 -num_iter_ft 5 -channel awgn -num_train_dec 5 -num_train_enc 1 -code_rate_k 1 -code_rate_n 3 -train_enc_channel_low 2.0 -train_enc_channel_high 2.0  -snr_test_start -1.5 -snr_test_end 4.0 -snr_points 12 -num_iteration 6 -is_parallel 1 -train_dec_channel_low -1.5 -train_dec_channel_high 2.0 -is_same_interleaver 1 -dec_lr 0.0001 -enc_lr 0.0001 -num_block 50000 -batch_size 500 -train_channel_mode block_norm -test_channel_mode block_norm -num_epoch 100 --print_test_traj -loss bce -init_nw_weight ./models/dta_cont_cnn2_cnn5_enctrain2_dectrainneg15_2.pt
    ```
