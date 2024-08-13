# Code for Replication of Context-aware Code Summary Generation

## Quick link
- [To-do list](#to-do-list)
- [Compiling dataset](#compiling-dataset)
- [Finetuning and Inference](#finetuning-and-inference)
- [Metrics](#metrics)

## To-do list
- To set up your local environment, run the following command. We recommend the use of a virtual environment for running the experiments.

```
pip install -r requirements.txt
```
- Please download the dataset from our [Hugginface](https://huggingface.co/datasets/apcl/jam_contextsum)
- Please download the pretrained model from our [Hugginface](https://huggingface.co/apcl/jam-contextsum)

## Compiling dataset

We release all of the raw data in our [Hugginface](https://huggingface.co/datasets/apcl/jam_contextsum). After you create ``bins``, ``pkls``, and ``tmp`` in ``data/method_holdout_context/context0``, you can simply run the following command to compile the data for 40 methods holdout automatically.

```
./compile_data_method_context.sh
```

## Finetuning and Inference
These steps will show you how to fine-tune the model for statement prediction.

### Step 1: Download the models for finetuning 
Please download the checkpoint files named ``ckpt_pretrain.pt`` in our [Hugginface](https://huggingface.co/apcl/jam-contextsum) for finetuning and place the checkpoint to the directory that you will copy from as in the ``execute_method.sh``.

### Step 2: Finetuning model
You can simply train the model for 40 methods holdout with the following command:

```
./execute_method.sh
```

## Metrics
We also provide the script for computing the automatic metrics. You can simply run the following command after you change the filename in the script.

```
./compute_automatic_metric.sh
```
To combine those metrics into a csv file, you can simply run the following command.

```
python3 compute_metrics.py --use-dir=metrics/leave-one-out/USE/gemini/pretrained/ --meteor-dir=metrics/leave-one-out/METEOR/gemini/pretrained/ --output-filename=metrics/leave-one-out/jam-pretrained-gemini.csv
```

- Related parameters are are as follows:
```
--use-dir: directory of the USE score file
--meteor: direcctory of the METEOR score file
--output-filename: output csv filename
```


