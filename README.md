# foundation_obj_nav
Object goal navigation with foundation models

## Installation
Add the submodules to the project:
```
git submodule update --init --recursive
``````

Install LAVIS:

```
pip install salesforce-lavis
```

Install GroundingDINO (run inside Grounded-Segment-Anything):

```
python -m pip install -e GroundingDINO
```

Install the requirements for Tag2Text (run inside Grounded-Segment-Anything):

```
cd Tag2Text && pip install -r requirements.txt
```

To get the baselines or sim setup running, you may need to install the `home_robot` package:
```
cd home-robot/src/home_robot && pip install -e .
```

Create folder to download checkpoints into, i.e. `mkdir checkpoints && cd checkpoints/`:
- RAM: `wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth`
- GroundingDINO: `wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`
- GNM (ViNT): `wget https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view?usp=drive_link`

*NOTE*: No environment setup prepared yet. When manually creating a Python environment, ensure that `transformers==4.26.1`. Older or newer version of `transformers` may cause various issues. To runn LLAVA, `transformers==4.31.0` is required.

*NOTE*: Several issues may be encountered while attempting to install and run `home_robot`:
- Failure to build `sophuspy`. Recommended resolution is to download and build `pybind11`, then use it to build `sophuspy` wheels: https://github.com/craigstar/SophusPy/issues/3.
- Missing dependency on `pytorch3d`. Recommended resolution is to build it from source, using the provided CUB library tar file. More details at: https://github.com/facebookresearch/pytorch3d/blob/3b4f8a4980e889936650e6841c6861ac45ed1117/INSTALL.md
- Missing dependency on `torch_geometric`. Recommended resolution is to `pip install torch_geometric`.

To run tests in habitat:

```
python navigate_homerobot.py
```

Folder structure:

```
├── data
│   ├── hm3d
│   │   │   ├── val
│   │   │   │   ├── content
│   │   │   │   ├── val.json.gz
│   ├── gibson
│   │   │   ├── v1.1
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   │   │   ├── content
│   │   │   │   │   ├── val.json.gz
│   │   │   │   │   ├── val_info.pbz2
│   ...
│   ├── logs
└── └──  ├── <name_of_trajN>
    	    ├── 0.jpg
    	    ├── 1.jpg
    	    ├── ...
            ├── action.txt
            └── query.log

```