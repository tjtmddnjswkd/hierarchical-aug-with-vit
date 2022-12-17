## DL2_project - Applying Hierarchical Augmentation to the DINO Method Using the ViT Model
This is a repository that implements [DINO](https://github.com/facebookresearch/dino) to apply for ViT the [paper](https://arxiv.org/abs/2206.00227)'s components which can be applied only CNN-based model.

<p align="left">
    <img width="400" alt="CNN+Hier" src="https://user-images.githubusercontent.com/69955858/208245712-846ce8be-c6b7-4fe1-af6e-2e7176cfdaf7.png">
</p>

<p align="right">
    <img width="400" alt="ViT(DINO)+Hier" src="!https://user-images.githubusercontent.com/69955858/208245659-ae098f04-08e9-41e3-a1ac-e18b8f95455d.png">
</p>

## Pretraining

The dataset used for pretraining is tinyimagenet200.

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

I use several fine-grained classification benchmarks to evaluate the learned representation with each method.

1. Simsiam, Simsiam+Hier

command:

```
python main_lincls.py -a resnet34 --dist-url 'tcp://localhost:10255' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/checkpoint --dataset [imagenet, cub, car, or flower]
```

2. DINO, DINO+Hier

command:

```
python -m torch.distributed.launch --nproc_per_node=2 eval_linear.py --dataset [imagenet, cub, car, or flower] --num_labels [200, 200, 196, or 102] --pretrained_weights /path/to/checkpoint
```

## Results
### Top-1 Acc.
<table>
  <tr>
    <th>Method</th>
    <th>TinyImageNet</th>
    <th>CUB</th>
    <th>CAR</th>
    <th>FLOWER</th>
  </tr>
  <tr>
    <td>SimSiam</td>
    <td>47.22</td>
    <td>21.51</td>
    <td>27.31</td>
    <td>63.63</td>
  </tr>
  <tr>
    <td>SimSiam+Hier</td>
    <td>49.66</td>
    <td>25.25</td>
    <td>26.27</td>
    <td>68.43</td>
  </tr>
  <tr>
    <td>DINO</td>
    <td>48.80</td>
    <td>25.65</td>
    <td>25.92</td>
    <td>65.20</td>
  </tr>
  <tr>
    <td>DINO+Hier</td>
    <td>47.03</td>
    <td>30.32</td>
    <td>24.65</td>
    <td>68.82</td>
  </tr>
</table>

### Top-5 Acc.

<table>
  <tr>
    <th>Method</th>
    <th>TinyImageNet</th>
    <th>CUB</th>
    <th>CAR</th>
    <th>FLOWER</th>
  </tr>
  <tr>
    <td>SimSiam</td>
    <td>72.22</td>
    <td>46.07</td>
    <td>51.15</td>
    <td>83.92</td>
  </tr>
  <tr>
    <td>SimSiam+Hier</td>
    <td>73.02</td>
    <td>50.54</td>
    <td>49.61</td>
    <td>86.77</td>
  </tr>
  <tr>
    <td>DINO</td>
    <td>74.26</td>
    <td>51.97</td>
    <td>48.50</td>
    <td>85.20</td>
  </tr>
  <tr>
    <td>DINO+Hier</td>
    <td>72.23</td>
    <td>57.58</td>
    <td>47.44</td>
    <td>87.67</td>
  </tr>
</table>
