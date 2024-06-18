# foundation_obj_nav
Object goal navigation with foundation models

## Installation (Simulation evaluations)
### OpenSearch setup
Create a conda environment (tested with Python 3.9).
```
conda create -n nav python=3.9 && \
conda activate nav
```
Install torch and torchvision (tested with PyTorch 2.3.1, using CUDA 11.8 libraries).
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Add Grounded-Segment-Anything submodule and required nested submodules to project.
```
git submodule update --init Grounded-Segment-Anything && \
cd Grounded-Segment-Anything && \
git submodule update --init Tag2Text
```
Install LAVIS for BLIP-2 and openai for GPT-3.5.
```
pip install salesforce-lavis && pip install openai
```
Install GroundingDINO.
```
python -m pip install -e GroundingDINO
```
Install Tag2Text.
```
cd Tag2Text && pip install -r requirements.txt
```
Change transformers back to original 4.26.1 version if it has been changed by Tag2Text.
```
pip install transformers==4.26.1
```
### Habitat simulation setup
Start install from root of repository.
Install torch_cluster (to pre-empt version conflicts if auto-installed by requirements later). Also install torch_geometric.
```
conda install pytorch-cluster -c pyg && pip install torch_geometric
```
Install habitat_sim (requires version 0.2.5)
```
conda install habitat-sim=0.2.5 withbullet headless -c conda-forge -c aihabitat
```
Install home_robot.
```
cd home-robot/src/home_robot && pip install -e .
```
Install home_robot_sim.
```
cd ../home_robot_sim && pip install -e .
```
Follow steps 2, 3 to install other dependencies required: https://github.com/facebookresearch/home-robot/tree/main/src/home_robot_sim.
Lastly, install scikit-fmm, if it has not yet been installed.
```
pip install scikit-fmm
```

Notes:
- `transformers==4.26.1` is required for packages to work together. Note that running LLaVa requires `transformers==4.31.0`.
- If there are issues building `sophuspy` for `home_robot`, recommended solution is to download and build `pybind11`, then build `sophuspy` wheels with it: https://github.com/craigstar/SophusPy/issues/3

### Organising repository and getting models
Create top-level folder to download checkpoints into, i.e. `mkdir checkpoints && cd checkpoints/`:
- RAM: `wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth`
- GroundingDINO: `wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`
- GNM (ViNT): `wget https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view?usp=drive_link`

Create top-level folders for logging and storing data: `mkdir logs && mkdir data`. Overall folder structure:
Folder structure:

```
├── data/scene_datasets
│   ├── hm3d
│   │   │   ├── val
│   │   │   │   ├── content
│   │   │   │   ├── val.json.gz
│   ├── gibson
│   │   │   ├── v1.1
│   │   │   │   ├── val
│   │   │   │   │   ├── content
│   │   │   │   │   ├── val.json.gz
│   │   │   │   │   ├── val_info.pbz2
├── checkpoints
│   ├── groundingdino_swint_ogc.pth
│   ├── ram_swin_large_14m.pth
├── configs
│   ├── gpt_config.yaml
│   ├── ...
├── ...
└── logs
    └──  <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── action.txt
        └── query.log

```
### Run simulation tests
```
python navigate_homerobot.py
```
