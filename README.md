# UGATIT_paddle
- 使用paddle复现论文UGATIT《Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation》，论文地址：
https://arxiv.org/abs/1907.10830
- 官方实现 https://github.com/taki0112/UGATIT（tensorflow版本） / https://github.com/znxlwm/UGATIT-pytorch（pytorch版本）

### 1. 数据的解压缩 与 环境处理
- 使用的数据集： 官方数据集 【selfie2anime】

```python
# 解压数据集
!mkdir dataset
!mkdir dataset/selfie2anime
!unzip -q data/data48600/selfie2anime.zip -d dataset/selfie2anime/
```

```python
# 升级 paddlepaddle-gpu
! pip install paddlepaddle-gpu==1.8.3.post97 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 训练模型
- 环境如果没有引入cuda路径，可以在前面加入 export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && 
```python
 python main.py --ch 64 --lr 0.0001 --result_dir results_ch64_01 --resume True --device cuda  --keep_result False --data_improve True
```

### 3. 测试模型
#### 1).在模型结果文件夹中会生成测试结果
- 测试集testA, testB 生成结果存储在 results_ch64_01/selfie2anime/test/single



```python
!export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && python main.py --phase test --ch 64 --lr 0.0001 --result_dir results_ch64_01 --resume True --device cuda  --keep_result False --data_improve True
```


#### 2).评估生成结果
- 下载生成结果，通过GAN_Metrics-Tensorflow(https://github.com/taki0112/GAN_Metrics-Tensorflow) 测试代码, 使用本地的Tensorflow进行验证：
	- selfie2anime 将 real_target文件夹存储testB动漫图片，fake文件夹存储为生成的A2B文件，进行测试
    	- 结果为 KID_mean :  8.507321029901505， KID_stddev :  0.5047781392931938
        
   - anime2selfie 将 real_target文件夹存储testA真人图片，fake文件夹存储为生成的B2A文件，进行测试
   		- 结果为 KID_mean :  5.521831661462784， KID_stddev :  0.4953019320964813

### 4. 模型文件
- 见 results_ch64_01/selfie2anime/model 下最大step 保存的文件

