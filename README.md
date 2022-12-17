## DL2_project - Applying Hierarchical Augmentation to the DINO Method Using the ViT Model
This is a repository that implements [DINO](https://github.com/facebookresearch/dino) to apply for ViT the [paper](https://arxiv.org/abs/2206.00227)'s components which can be applied only CNN-based model.

## Results

<table>
  <tr>
    <th>Method</th>
    <th>Pretraining dataset</th>
    <th colspan="2", rowspan="2">TinyImageNet</th>
  </tr>
</table>

## Pretraining

1. SimSiam

path: /simsiam/
command: 

```
python main_simsiam.py \
  -a resnet34 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  --data /path/to/tinyimagenet/
```

2. SimSiam+Hier

path: /simsiam+hier/
command: 

```
python main_train.py \
  -a resnet34 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  --data /path/to/tinyimagenet
```

3. DINO

path: /dino/
command: 

```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /path/to/tinyimagenet/train --output_dir /path/to/saving_dir
```

4. DINO+Hier

path: /dino/
command: 

```
python -m torch.distributed.launch --nproc_per_node=8 main_dino_hier.py --arch vit_small --data_path /path/to/tinyimagenet/train --output_dir /path/to/saving_dir
```

## Linear evaluation

