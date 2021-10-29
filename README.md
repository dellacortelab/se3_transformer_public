# se3_transformer

This is a reimplementation of the original SE(3)-Transformer https://github.com/FabianFuchsML/se3-transformer-public.

## Getting Started

To download this repo, run

    git clone https://github.com/dellacortelab/se3_transformer_public.git

To setup the environment for this repo, it is recommended that you install the latest version of anaconda and use the yml file to install the depencies by running the command

    conda env create -f se3.yml
  
You'll also want the home directory to your python path so that the code can find everything it needs:

    export PYTHONPATH=$PYTHONPATH:$(pwd)/se3_transformer
    
The ANI1x dataset can be downloaded with

    wget https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/18112775/ani1xrelease.h5 -P ./experiment/

## Using this repo

This repo can be used to train, validate, and test SE(3)-Transformer models.  To train a basic model, run:

    python ./experiment/train.py

To run validation on models saved during training, run

    python ./experiment/validation.py --model_dir ./experiment/models/basic

To run inference on the test set on a model, say modeL_1000.pkl, run

    python ./experiment/test.py --model_dir ./experiment/models/basic/model_1000.pkl
