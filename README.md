# InTraVisTo

## Abstract

> The reasoning capabilities of Large Language Models (LLMs) have increased greatly over the last few years, as has their size and complexity. Nonetheless, the use of LLMs in production remains challenging due to their unpredictable nature and discrepancies that can exist between their desired behaviour and their actual model output.
> In this paper, we introduce a new tool, **InTraVisTo** (Inside Transformer Visualization Tool), designed to enable researchers to investigate and trace the computational process that generates each token in a Transformer-based LLM. 
> InTraVisTo provides a visualization of both the *internal state* of the Transformer model (by decoding token embeddings at each layer of the model) and the *information flow* between the various components across the different layers of the model (using a Sankey diagram). 
> With InTraVisTo, we aim to help researchers and practitioners better understand the computations being performed within the Transformer model and thus to shed some light on internal patterns and reasoning process employed by LLMs.

---

## Installation

To begin, clone the repository to your local machine using the following commands:
```bash
git clone --recurse-submodules https://github.com/daviderigamonti/InTraVisTo.git
cd InTraVisTo
```
Afterwards, you can either choose to directly execute *InTraVisTo* [from the Python source code](#execute-from-the-source-code), or use the [dockerized version](#execute-using-docker).

### Execute using Docker

#### Prerequisites

In order to execute *InTraVisTo* using docker you will need to have [Docker](https://www.docker.com/) installed on your system.
Additionally, in order to build the docker image you will also need the [Buildx](https://docs.docker.com/build/architecture/) CLI tool, which should automatically be installed along the [Docker Desktop](https://www.docker.com/products/docker-desktop/) application.
> :information_source: Depending on your system and your installation, administrator privileges might be needed to run the `docker` commands present in this guide.

A [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens) with READ privileges is also needed as a way to interact with the Hugging Face API and download models to use with *InTraVisTo*.
It can freely be obtained from the [Hugging Face profile settings page](https://huggingface.co/settings/tokens) if you have an account.
> :information_source: If you are utilizing your personal Hugging Face token, some models may require a confirmation step in order to be granted access to their weights. This can usually be done directly from the model's page on [Hugging Face :hugs:](https://huggingface.co/).

#### Process

You will first need to build the docker image according to the [`Dockerfile`](/Dockerfile), using the following command:
```bash
docker buildx build -t intravisto .
```
This will create a docker image called `intravisto`, that will be stored by docker on your system.

In order to actually run the docker image, you will need to run the following command:
```bash
sudo docker run --rm --runtime=nvidia --gpus all -d -p 8892:<PORT> -e HF_TOKEN=<TOKEN> -v $(pwd)/huggingface:/app/huggingface:rw --name intravisto intravisto
```
The `<PORT>` and `<TOKEN>` placeholders indicate respectively the host port that you wish to expose *InTraVisTo* from, and your Hugging Face token in order to access the repositories for various models on [Hugging Face :hugs:](https://huggingface.co/).
A notable part of this command is the GPU runtime settings (represented by the `--runtime=nvidia --gpus all` portion), which needs to be customized according to the GPU availability of the host machine.
Another important section is the persistent memory mount point (indicated by `-v $(pwd)/huggingface:/app/huggingface:rw`) that defines the storage location of cached huggingface models inside the host machine filesystem; by default this is set to be the `huggingface` directory inside the cloned copy of this repository.

### Execute from the Source Code

#### Prerequisites

A version of [Python](https://www.python.org/) `>= 3.11` should be already installed on your host system.
> :warning: For simplicity, all commands will reference the python system installation.
> However, the use of a [Python virtual environment](https://docs.python.org/3/library/venv.html) is strongly recommended.

A [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens) with READ privileges is also needed as a way to interact with the Hugging Face API and download models to use with *InTraVisTo*.
It can freely be obtained from the [Hugging Face profile settings page](https://huggingface.co/settings/tokens) if you have an account.
> :information_source: If you are utilizing your personal Hugging Face token, some models may require a confirmation step in order to be granted access to their weights. This can usually be done directly from the model's page on [Hugging Face :hugs:](https://huggingface.co/).

#### Process

The first step for locally executing *InTraVisTo* directly from the source code, is to download its dependencies using `pip`.
All direct dependencies are listed inside the [`requirements file`](/requirements.txt), however submodules' dependencies will also need to be installed.
To do so, it's sufficient to execute the following command inside *InTraVisTo*'s root directory:
```bash
pip install -r requirements.txt -r ./submodules/transformer_wrappers/requirements.txt
```

The next step consists of adding the submodules dependencies to the `PYTHONPATH` environment variable:
```bash
export PYTHONPATH=${PYTHONPATH}:./submodules/transformer_wrappers/src
```
It is possible to include this command inside the `venv/bin/activate` file if you are working inside a dedicated Python virtual environment and would like to perform this operation every time the environment is loaded, otherwise it needs to be done manually.

As a last step, the Hugging Face token also needs to be added as an environment variable, in order to access the repositories for various models on [Hugging Face :hugs:](https://huggingface.co/):
```bash
export HF_TOKEN=<TOKEN>
```

At this point, you can execute *InTraVisTo* as:
```bash
python ./src/InTraVisTo.py
```
This should create a local server on your host system on port `8892` by default.
The application should be reachable at the following URL: [`http://0.0.0.0:8892/intravisto/`](http://0.0.0.0:8892/intravisto/).

---

## Authors

- Nicolò Brunello
- Davide Rigamonti
- Vincenzo Scotti
- Mark James Carman
  
> DEIB, Politecnico di Milano
> 
> Via Ponzio 34/5, 20133, Milano (MI), Italy

---

## Acknowledgments

- [Transformer Wrappers Library](https://github.com/vincenzo-scotti/transformer_wrappers)
- [Hugging Face :hugs:](https://huggingface.co/)
- [Plotly Dash](https://dash.plotly.com/)
