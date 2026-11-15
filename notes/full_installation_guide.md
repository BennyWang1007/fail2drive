
# ================= One terminal =================
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y gcc-11 g++-11
su - ubuntu

cd ~

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3


## HiP-AD
```bash
git clone https://github.com/BennyWang1007/HiP-AD-steering.git HiP-AD
cd HiP-AD

source ~/.bashrc
conda create -n hipad python=3.8 -y
conda activate hipad
pip install --upgrade pip

conda install -y -c "nvidia/label/cuda-11.7.0" -c conda-forge cuda-toolkit

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export NVCC_CCBIN=/usr/bin/g++-11

pip install -r requirements.txt

cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../../

mkdir -p ckpts
cp projects/configs/hipad_b2d_stage2.py ckpts/
wget -P ckpts https://github.com/nullmax-vision/HiP-AD/releases/download/weighs/hipad_stage2.pth

# After installing fail2drive, run these

conda deactivate && conda activate hipad
pip install -r ~/fail2drive/team_code/requirements_hipad.txt

```

## fail2drive
```bash

# ================= Another terminal =================

cd ~
git clone https://github.com/BennyWang1007/fail2drive.git
cd fail2drive

mkdir f2d_carla
curl -L \
  https://huggingface.co/datasets/SimonGer/fail2drive/resolve/main/fail2drive_simulator.tar.gz \
  | tar -xz -C f2d_carla

source ~/.bashrc

conda env create -f environment.yml
conda activate fail2drive

pip install jaxtyping==0.3.2
pip install beartype==0.21
pip install numba
pip install torchmetrics
conda install -y -c conda-forge ffmpeg

source env_vars.sh

conda deactivate && conda activate fail2drive
```

## Depth Anything V2 (continue with conda env fail2drive)
```bash
conda activate fail2drive

cd ~
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2

pip install -r requirements.txt

mkdir -p ~/Depth-Anything-V2/metric_depth/checkpoints
cd ~/Depth-Anything-V2/metric_depth/checkpoints

pip install huggingface_hub
hf auth login

# Enter your Hugging Face credentials when prompted.

hf download \
  depth-anything/Depth-Anything-V2-Metric-VKITTI-Large \
  depth_anything_v2_metric_vkitti_vitl.pth \
  --local-dir ~/Depth-Anything-V2/metric_depth/checkpoints \
  --type model
```

## lead-cvpr2026
```bash
# ================= Another terminal =================
cd ~
git clone https://github.com/BennyWang1007/lead-steering.git lead-cvpr2026
cd lead-cvpr2026

echo -e "export LEAD_PROJECT_ROOT=$(pwd)" >> ~/.bashrc
echo "source $(pwd)/scripts/main.sh" >> ~/.bashrc

bash scripts/download_one_checkpoint.sh
```

