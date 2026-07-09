# Prefix RIPO-like Delta Sweep

这个目录保存 `prefix_ripo_clip` 的 delta sweep 任务配置。当前实验使用
Qwen3-4B-Base、processed DAPO-Math-17k 训练集和 AIME24 验证集，其他训练
参数与 GSPO baseline 对齐，只改变 clip 策略和 `RIPO_DELTA_LOW/HIGH`。

## 环境镜像

当前平台提交使用的镜像为：

```text
gensi-cn-beijing.cr.volces.com/sia-thu/verl:v0
```

外场平台如果支持直接填写镜像，使用同一个镜像即可。该镜像内置 conda 环境，
脚本不会强制激活仓库下的 `.venv`。

## 克隆仓库并切换分支

```bash
git clone https://github.com/LiChengxi666/verl.git
cd verl
git checkout package-prefix-ripo-recipes
```

## 准备模型和数据

默认路径都相对于仓库根目录：

```text
models/Qwen3-4B-Base/
data/data_processed/math-17k.parquet
data/data_processed/aime24.parquet
```

模型可以用 Hugging Face CLI 下载到默认位置：

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen3-4B-Base \
  --local-dir models/Qwen3-4B-Base \
  --local-dir-use-symlinks False
```

processed 数据需要放到：

```bash
mkdir -p data/data_processed
# 将 math-17k.parquet 和 aime24.parquet 放到 data/data_processed/ 下
```

如果模型或数据放在其他路径，运行前覆盖环境变量即可：

```bash
export MODEL_PATH=/path/to/Qwen3-4B-Base
export TRAIN_FILE=/path/to/math-17k.parquet
export VAL_FILE=/path/to/aime24.parquet
```

## 一键启动 RIPO-like 默认实验

在一台 8 卡机器的仓库根目录下运行：

```bash
LOSS_MODE=prefix_ripo_clip \
EXPERIMENT_NAME=prefix-ripo-l1em5-h3em5-qwen3-4b-8gpu \
RIPO_DELTA_LOW=1e-5 \
RIPO_DELTA_HIGH=3e-5 \
bash scripts/experiments/run_qwen3_math_rl.sh
```

这条命令会默认写入：

```text
checkpoints/verl_math_repro/<experiment_name>/
train_logs/verl_math_repro/<experiment_name>/
validation_generations/verl_math_repro/<experiment_name>/
```

训练脚本会自动开启 checkpoint resume：

```text
trainer.resume_mode=auto
trainer.save_freq=10
trainer.test_freq=10
```

如果设置了 `WANDB_API_KEY`，或设置了可读的 `WANDB_API_KEY_FILE`，脚本会自动
把 `wandb` 加入 logger；否则只使用 `console,file,tensorboard`。

## Delta Sweep

这里的 `prefix_ripo_delta_low/high` 是 average-token prefix KL budget。不要直接
复用旧的 `0.02/0.05/0.08` 试验值，除非你明确想测试一个非常松的 trust region。

当前 sweep：

```text
1e-7 / 3e-7
3e-7 / 1e-7
1e-6 / 3e-6
3e-6 / 1e-6
1e-5 / 3e-5
3e-5 / 1e-5
1e-4 / 3e-4
3e-4 / 1e-4
```

在当前 Volc/MLP 平台上批量提交：

```bash
bash training_jobs/scripts/submit_prefix_ripo_delta_sweep.sh
```

这些 YAML 仍包含平台资源、队列和存储字段；迁移到其他平台时，需要按目标平台
修改 `ResourceQueueID/ResourceQueueName`、实例规格、TensorBoard 存储和数据挂载。
当前 YAML 假设仓库内容挂载在 `/workspace`，因此入口命令会先 `cd /workspace`；
如果目标平台使用其他挂载点，请同步修改 YAML 中的 `MountPath` 和 `Entrypoint`
里的 `cd` 路径。训练脚本本身会从脚本位置反推仓库根目录，不再依赖个人目录。
