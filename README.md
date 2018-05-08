# python3_tf_slim_image_classify
Python3使用TF-Slim进行图像分类

# 机器环境
- win10
- python3.6
- tensorflow==1.7.0

# 准备图片数据
- 准备好自定义的图片数据
- 放到 data_prepare/pic/train 和 data_prepare/pic/validation 中
- 自己建立分类文件夹，文件夹名为分类标签名


# 将图片数据转换成TF-Record格式文件
- 在 data_prepare/ 下，执行

```python
python data_convert.py -t pic/ \
  --train-shards 2 \
  --validation-shards 2 \
  --num-threads 2 \
  --dataset-name satellite
```

- 会生成4个tf-record文件和1个label文件


# 将转换生成的5个文件复制到 slim\satellite\data 下

# 下载预训练模型Inception V3 
- http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
- 解压后，复制到 slim\satellite\pretrained 下


# 在 slim/ 文件夹下执行如下命令，进行训练：

```python
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=2 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

# 在 slim/ 文件夹下执行如下命令，进行模型能力评估：

```python
python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_v3
```

# 导出训练好的模型
- 在 slim/ 文件夹下面执行如下命令：

```python
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=satellite/inception_v3_inf_graph.pb \
  --dataset_name satellite
```

- 在 项目根目录 执行如下命令（需将5271改成train_dir中保存的实际的模型训练步数）

```python
python freeze_graph.py \
  --input_graph slim/satellite/inception_v3_inf_graph.pb \
  --input_checkpoint slim/satellite/train_dir/model.ckpt-5271 \
  --input_binary true \
  --output_node_names InceptionV3/Predictions/Reshape_1 \
  --output_graph slim/satellite/frozen_graph.pb
```

# 对单张图片进行预测
- 在 项目根目录 执行如下命令

```python
python classify_image_inception_v3.py \
  --model_path slim/satellite/frozen_graph.pb \
  --label_path data_prepare/pic/label.txt \
  --image_file test_image.jpg
```