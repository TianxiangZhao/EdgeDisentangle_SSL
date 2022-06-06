# EdgeDisentangle_SSL
Pytorch implementation of paper ['Exploring edge disentanglement for node classification'](https://dl.acm.org/doi/pdf/10.1145/3485447.3511929) on WWW2022.

## Dataset
Four dataset, cora, cora_full, and chameleon are included in this directory.

## Configurations

### Models
- Our proposed model can be selected as '--model=DISGAT'. A set of baseline architectures, including GCN, GraphSage, Mixhop and GIN are also provided.
- For DISGAT, 'gnn_type' selectes which GNN layer to use inside DISGAT block, including AT, SAGE, and GCN. 
- For DISGAT, '--att' controls with attention mechanism to use.

### SSL tasks
- Disentanglement-encouraging signals, including supervision on edge recovery (SupEdge), homo/hetero edges (DisEdge), and head diversity (DifHead) are implemented.
- SSL tasks can be selected with '--pretrain [*task1,task2,...*] --pre_weight [*weight1,weight2,...*] --pre_edge [*edge1,edge2,...*]'. '--pre_weight' sets the weight of corresponding SSL task, and '--pre_edge' selects which edge set to use (Only implemented for heterogeneous graphs, can just set it to 1 for most datasets) 
- For SSL task-specific configuration terms, please refer to *utils.py* and *pretainer.py*.
- '--constrain_layer' sets the layers to which SSL tasks are applied.

### Downstream task
- Please set it as: '--downstream CLS --down_weight 1.0 --finetune'. 
- We tried edge prediction as another downstream task, but it is unfinished yet. 
- Without '--finetune', the GNN will only be trained on SSL tasks for pre-training.

## Example
An example of running DISGAT on cora_full dataset is provided.


If any problems occurs via running this code, please contact us at tkz5084@psu.edu.

Thank you!
