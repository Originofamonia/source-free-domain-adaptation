## Get and prepare the practice data

The trial data for the practice phase consists of 99 articles from the _AQUAINT_, _TimeBank_ and _te3-platinum_ subsets of _TempEval-2013_, i.e. _"Newswire"_ domain.

You can automatically download and prepare the input data for this phase running the `prepare_time_dataset.py` script available in the [task repository](https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation). If you don't already have the task repo checked out and the requirements installed, you need to do so first:

$ git clone https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation.git && cd source-free-domain-adaptation

$ pip3 install -r baselines/time/requirements.txt

$ python3 prepare\_time\_dataset.py practice\_text/

This will create a `practice_text/time` directory containing the plain text of the [documents used in this task](https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation/blob/master/practice_time_documents.txt).

## Get the model and make predictions on the practice data

The baseline for the time expression recognition is based on the pytorch implementation of [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) by _Hugging Face_. We have used the `[RobertaForTokenClassification](https://huggingface.co/transformers/model_doc/roberta.html?#robertafortokenclassification)` architecture from _Hugging Face/transformers_ library to fine-tune `[roberta-base](https://huggingface.co/roberta-base)` on 25,000+ time expressions in de-identified clinical notes. The resulting model is a sequence tagger that we have made available in _Hugging Face_ model hub: [clulab/roberta-timex-semeval](https://huggingface.co/clulab/roberta-timex-semeval). The following table shows the _in-domain_ and _out-of-domain (practice\_data)_ performances:

P

R

F1

in-domain\_data

0.967

0.968

0.968

practice\_data

0.775

0.768

0.771

The task repository contains scripts to load and run the model: `[time baseline](https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation/tree/master/baselines/time)`. These scripts are based on the _Hugging Face/transformers_ library that allows easily incorporating the model into the code. See for example, [the code from the baseline that loads the model and its tokenizer](https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation/blob/master/baselines/time/run_time.py#L21).

The first time you run such code, the model will be automatically downloaded in your computer. The scripts also include the basic functionality to read the input data and produce the output _Anafora_ annotations. You can use the `run_time.py` script to parse raw text and obtain time expressions. For example, to process the practice data, run:

  
$ python3 baselines/time/run\_time.py -p practice\_text/time/ -o submission/time/

This will create one directory per document in `submission/time` containing one `.xml` file with predictions in _Anafora_ format.

## Extend the baseline model

There are many ways to try to improve the performance of this baseline on the practice text (and later, on the evaluation text). Should you need to continue training the `clulab/roberta-timex-semeval` model on annotated data that you have somehow produced, you can run the `train_time.py` script:

$ python3 baselines/time/train\_time.py -t /path/to/train-data -s /path/to/save-model

The `train-data` directory must follow a similar structure to the `practice_text/time` folder and include, for each document, a the raw text file (with no extension) and an _Anafora_ annotation file (with `.xml` extension). After running the training, the `save-model` directory will contain three files (`pytorch_model.bin`, `training_args.bin` and `config.json`) with the configuration and weights of the final model, and the vocabulary and configuration files used by the tokenizer (`vocab.json`, `merges.txt`, `special_tokens_map.json` and `tokenizer_config.json`).