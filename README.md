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

Create folder to download checkpoints into, i.e. `mkdir checkpoints && cd checkpoints/`:
- RAM: `wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth`
- GroundingDINO: `wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`
- GNM (ViNT): `wget https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view?usp=drive_link`

NOTE: No environment setup prepared yet. When manually creating a Python environment, ensure that `transformers==4.26.1`. Older or newer version of `transformers` may cause various issues.