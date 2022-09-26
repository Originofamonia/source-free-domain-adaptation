# Overview

## Welcome

This is the CodaLab Competition for SemEval-2021 Task 10: Source-Free Domain Adaptation for Semantic Processing.

Please join our [Google Group](https://groups.google.com/forum/#!forum/source-free-domain-adaptation-participants/) to ask questions and get the most up-to-date information on the task.

### Important Dates:

31 Jul 2020:

  Trial data release

4 Sep 2020:

  Training data release

3 Dec 2020:

  Test data release

10 Jan 2021:

  Evaluation start

31 Jan 2021:

  Evaluation end

## Summary

Data sharing restrictions are common in NLP datasets. For example, Twitter policies do not allow sharing of tweet text, though tweet IDs may be shared. The situtation is even more common in clinical NLP, where patient health information must be protected, and annotations over health text, when released at all, often require the signing of complex data use agreements. The SemEval-2021 Task 10 framework asks participants to develop semantic annotation systems in the face of data sharing constraints. A participant's goal is to develop an accurate system for a target domain when annotations exist for a related domain but cannot be distributed. Instead of annotated training data, participants are given a model trained on the annotations. Then, given unlabeled target domain data, they are asked to make predictions.

## Tracks

We propose two different semantic tasks to which this framework will be applied: negation detection and time expression recognition.

*   **Negation detection** asks participants to classify clinical event mentions (e.g., diseases, symptoms, procedures, etc.) for whether they are being negated by their context. For example, the sentence: _Has no diarrhea and no new lumps or masses_ has three relevant events (diarrhea, lumps, masses), two cue words (both _no_), and all three entities are negated. This task is important in the clinical domain because it is common for physicians to document negated information encountered during the clinical course, for example, when ruling out certain elements of a differential diagnosis. We expect most participants will treat this as a "span-in-context"' classification problem, where the model will jointly consider both the event to be classified and its surrounding context. For example, a typical transformer-based encoding of this problem for the _diarrhea_ event in the example above looks like: _Has no <e>diarrhea</e> and no new lumps or masses_.
*   **Time expression recognition** asks participants to find time expressions in text. This is a sequence-tagging task that will use the fine-grained time expression annotations that were a component of SemEval 2018 Task 6 (Laparra et al. 2018). For example:  
      
    ![](https://i.ibb.co/tbmjZkh/time-example.png)  
      
    We expect most participants will treat this as a sequence classification problem, as in other named-entity tagging tasks.

## Organizers

Egoitz Laparra, Yiyun Zhao, Steven Bethard (University of Arizona)

Tim Miller (Boston Children's Hospital and Harvard Medical School)

Özlem Uzuner (George Mason University)

## References

Laparra E., Xu D., Elsayed A., Bethard S., and Palmer M. [SemEval 2018 task 6: Parsing time normalizations.](https://www.aclweb.org/anthology/S18-1011.pdf) In Proceedings of The 12th International Workshop on Semantic Evaluation, New Orleans, Louisiana. 2018.