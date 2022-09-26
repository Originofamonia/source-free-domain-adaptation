# Models

Paricipants are provided with trained models for both negation detection and time expression recognition. In both cases, we have used the _RoBERTa-base_ (Liu et al., 2019) pretrained model included in the [Huggingface Transformers](https://github.com/huggingface/transformers) library:

*   For **negation detection**, we provide a "span-in-context" classification model, fine-tuned on the 10,259 instances (902 negated) in the SHARP Seed dataset of de-identified clinical notes from Mayo Clinic, which the organizers have access to but cannot currently be distributed (models are approved to be distributed). In the SHARP data, clinical events are marked with a boolean polarity indicator, with values of either _asserted_ or _negated_.
*   For **time expression recognition**, we provide a sequence tagging model, fine-tuned on the 25,000+ time expressions in the de-identified clinical notes from the Mayo Clinic in SemEval 2018 Task 6, which are available to the task organizers, but are difficult to gain access to due to the complex data use agreements necessary (models are approved to be distributed).

## References

Liu Y., Ott M., Goyal N., Du J., Joshi M., Chen D., Levy O., Lewis M., Zettlemoyer L., and Stoyanov V. [Roberta: A robustly optimized bert pretraining approach](https://arxiv.org/pdf/1907.11692.pdf). arXiv preprint. 2019.