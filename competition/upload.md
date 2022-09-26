# Uploading predictions to CodaLab

To upload your predictions to CodaLab, first make sure that your predictions are formatted correctly, then create a `submission.zip` and upload it to CodaLab.

## Formatting system output

For **negation detection**, the output format is one classifier output per line, where the lines correspond to the lines in the input. A prediction of "Negated" should be output as 1, while a prediction of "Not negated" should be output as -1.

For **time expression recognition**, your system must produce [Anafora XML format](https://github.com/weitechen/anafora) files in Anafora's standard directory organization.

Make sure that you comply with following rules when you create  your output directory:

*   The root must contain only the track directories, `negation` and `time`. If you are not participating in one of the tracks, do not include its directory.
*   In the `negation` directory, include a single tsv file with the name `system.tsv`.
*   The `time` directory, follow the same structure and names as in the dataset:

*   Each top-level directory must contain only document directories, named exactly as in the input dataset.
*   Each document directory must contain only the corresponding annotation file.
*   The name of each annotation file must match the document name and have a `.TimeNorm.system.completed.xml` extension.

For example, for the development data, your directory structure should look like:

*   negation

*   system.tsv

*   time

*   AQUAINT

*   APW19980807.0261

*   APW19980807.0261.TimeNorm.system.completed.xml

*   APW19980808.0022

*   APW19980808.0022.TimeNorm.system.completed.xml

*   ...

*   TimeBank

*   ABC19980108.1830.0711

*   ABC19980108.1830.0711.TimeNorm.system.completed.xml

*   ABC19980114.1830.0611

*   ABC19980114.1830.0611.TimeNorm.system.completed.xml

*   ...

## Generating and uploading submission.zip

The easiest way to generate `submission.zip` is to use the `Makefile` provided in the [sample code repository](https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation/). First, place your prediction files - including the entire directory structure described above - under a `submission` directory in the root of the sample code checkout. Then run `make submission.zip`. This will zip up all your prediction files and produce a file, `submission.zip`.

To upload your submission, go to the CodaLab competition page. Find the "Participate" tab, then the "Submit/View Results" navigation element, then make sure "Practice" button is highlighted, and click the "Submit" button. Find your `submission.zip` with the file chooser and upload. The scoring will run in the background -- usually you can refresh the page in about a minute to see the result in the table below.

## Troubleshooting

You may see the error:

Traceback (most recent call last):
  File "/worker/worker.py", line 330, in run
    if input\_rel\_path not in bundles:
TypeError: argument of type 'NoneType' is not iterable

This is a [known issue with CodaLab](https://github.com/codalab/codalab-competitions/issues/2814#issuecomment-664142717). The solution for now is to make a new submission with the same `submission.zip`.