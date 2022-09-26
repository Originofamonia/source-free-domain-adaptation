.results {border-collapse: collapse;} .results th, td {padding: 5px;} .results th {background-color: #f2f2f2;} .results tr:nth-child(even) {background-color: #f2f2f2;} .results tr:nth-child(odd) {background-color: #ffffff;} .codestyle {background-color: #f2f2f2; padding: 3px; color: #666666;} .prestyle {background-color: #f2f2f2; padding: 5px; padding-left: 30px;}

# Getting Started: Negation

## Get the unlabeled development data

The practice data (development data) is a subset of the i2b2 2010 Challenge on concepts, assertions, and relations in clinical text. If you do not already have access to this data, you will need to request access at the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). If you have obtained access, follow the portal link above and download the data by expanding the "2010 Relations Challenge Downloads" tab, and downloading the three files with the following titles:

*   Training Data: Concept assertion relation training data
*   Test Data: Reference standard for test data
*   Test Data: Test data

At the time of writing, these are the last 3 links for the 2010 data. This should give you the following files, which you should save to a single directory:

*   concept\_assertion\_relation\_training\_data.tar.gz
*   reference\_standard\_for\_test\_data.tar.gz
*   test\_data.tar.gz

Extract each of these with:

*   `tar xzvf concept_assertion_relation_training_data.tar.gz`
*   `tar xzvf reference_standard_for_test_data.tar.gz`
*   `tar xzvf test_data.tar.gz`

Next we will extract an unlabeled training set, unlabeled evaluation set, and a label file for the evaluation set (to test submissions and see the format). If you don't already have the task repo checked out, do so and enter the project directory:

$ git clone https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation.git && cd source-free-domain-adaptation

Then to extract the training files, run the i2b2 extraction script with:

  
$ mkdir -p practice\_text/negation && python3 extract\_i2b2\_negation.py <directory with three extracted i2b2 2010 folders> practice\_text/negation

This will extract the three files into `practice_text/negation`:

*   train.tsv -- the unlabeled training data
*   dev.tsv -- the unlabeled deveopment data
*   dev\_labels.txt -- the labels for dev data

The idea during the practice time is to use train.tsv as representative target-domain data to improve the system, and then evaluate any improvements to your system on dev.tsv.

## Get the pretrained model and make predictions

To use the trained model to make predictions, install the requirements and run the `run_negation.py` script to process the practice data as follows:

$ pip3 install -r baselines/negation/requirements.txt
$ python3 baselines/negation/run\_negation.py -f practice\_text/negation/dev.tsv -o submission/negation/

This script will write a file called `submission/negation/system.tsv` with one label per line.