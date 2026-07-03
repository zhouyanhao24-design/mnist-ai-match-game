# MNIST AI Match Game

基于 MNIST 手写数字识别模型的三消游戏课程设计项目。

## 功能

- 使用 ResNet 风格模型完成 MNIST 分类。
- 使用 VGG 风格模型完成 MNIST 分类。
- 将训练好的模型接入三消游戏。
- 支持人为设置预测错误率，用于观察识别精度对游戏过程的影响。
- 保存 AI 辅助编程记录、Bug 记录和实验记录。

## 环境配置

```bash
conda create -n mnist-game python=3.10 -y
conda activate mnist-game
pip install -r requirements.txt
```

## 训练模型

```bash
python src/train.py --model resnet --epochs 5 --batch-size 128 --lr 0.001
python src/train.py --model vgg --epochs 5 --batch-size 128 --lr 0.001
```

## 评估模型

```bash
python src/evaluate.py --model resnet --checkpoint checkpoints/resnet_mnist.pth
python src/evaluate.py --model vgg --checkpoint checkpoints/vgg_mnist.pth
```

## 运行游戏

```bash
python src/game.py --model resnet --checkpoint checkpoints/resnet_mnist.pth
python src/game.py --model vgg --checkpoint checkpoints/vgg_mnist.pth
```

设置 20% 随机预测错误率：

```bash
python src/game.py --model resnet --checkpoint checkpoints/resnet_mnist.pth --error-rate 0.20
```

隐藏原始图片：

```bash
python src/game.py --model resnet --checkpoint checkpoints/resnet_mnist.pth --hidden
```

## 游戏规则

玩家点击 MNIST 图片方块后，程序调用模型预测数字类别。预测结果进入下方卡槽。卡槽中相同数字达到 3 个时自动消除。卡槽满且不能消除时失败，所有方块消除时胜利。

## 报告材料

`reports/` 目录包含 AI 提示词记录、Bug 修复记录和实验结果记录，可作为课程设计报告素材。
