# trocr-bullinger-htr
Use of Microsoft's [TrOCR](https://arxiv.org/abs/2109.10282), an encoder-decoder model consisting of an image Transformer encoder and a text Transformer decoder for state-of-the-art optical character recognition (OCR) and handwritten text recognition (HTR) on text line images. 

This particular model was fine-tuned on [Bullinger Dataset](https://github.com/pstroe/bullinger-htr) as part of the project [Bullinger Digital](https://www.bullinger-digital.ch) ([References](https://www.cl.uzh.ch/de/people/team/compling/pstroebel.html#Publications)).

* HF `model card`: [pstroe/bullinger-general-model](https://huggingface.co/pstroe/bullinger-general-model) | [Flexible Techniques for Automatic Text Recognition of Historical Documents](https://doi.org/10.5167/uzh-234886)

## Create an environment
    conda create -n trocr-bullinger python=3.9

## Install the requirements
    conda activate trocr-bullinger
    pip install -r requirements.txt

## Run
    python app.py
