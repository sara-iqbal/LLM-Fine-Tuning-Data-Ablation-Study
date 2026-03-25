# LLM Fine-Tuning & Data Ablation Study

This project came out of a simple question I kept seeing in ML papers but never actually doing myself does the amount of data you train on matter more, or does the quality of that data matter more? And what about how the data is balanced across classes? I wanted to find out properly, not just theoretically.

So I took DistilBERT, fine-tuned it on a financial sentiment dataset, and ran three structured experiments to test exactly that.

**Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1DGMvOKl2-Ljgnmzn6I5vsniGF9vENo4W#scrollTo=3uBfITS97P1y)

**Live Dashboard:** [sara-iqbal.github.io/LLM-Fine-Tuning-Data-Ablation-Study](https://sara-iqbal.github.io/LLM-Fine-Tuning-Data-Ablation-Study/)

---

## The Dataset

I used `financial_phrasebank` — specifically the `sentences_allagree` split, which only includes sentences where every single human annotator agreed on the label. That means the starting data is as clean as it gets. Three classes: negative, neutral, positive financial sentiment.

---

## What I Actually Did

### Baseline First

Before running any experiments, I trained a full model on the complete dataset for 3 epochs. This gave me a performance ceiling to measure everything else against. The baseline ended up around 88-90% accuracy, which is a solid reference point.

### Experiment 1 — How Much Data Do You Actually Need?

I trained six separate models, each on a different slice of the training data: 10%, 20%, 40%, 60%, 80%, and 100%. Same model, same settings, just less data each time.

What I found was that the model got surprisingly capable at 20% of the data. Going from 10% to 20% gave a massive jump in performance, but after 40% the gains started flattening out. This is the classic diminishing returns curve — and seeing it happen in practice on your own experiment is different from just reading about it.

The takeaway: you don't always need more data, you need enough data. There's a point where collecting more isn't worth the effort.

### Experiment 2 — Does Label Quality Matter?

I set up three versions of the training set:

- The original clean labels (all annotators agreed)
- A noisy version where I randomly flipped 20% of the labels
- A length-filtered version where I removed very short and very long sentences

The noisy condition was the most instructive. Even flipping just 20% of labels caused a meaningful drop in F1 score. The model still learned something, but it learned the wrong patterns in places. This confirmed something I suspected — label quality often matters more than raw data volume.

The length-filtered version performed comparably to clean, which suggests that extreme-length sentences add noise of their own kind.

### Experiment 3 — Does Class Balance Matter?

The original dataset isn't perfectly balanced — neutral sentences dominate. I tested three recipes:

- A perfectly balanced 1:1:1 mix
- A neutral-heavy 1:3:1 mix
- The original natural distribution

The neutral-heavy mix hurt per-class recall on negative and positive, which makes sense — the model sees fewer examples of minority classes and becomes less confident about them. The balanced mix was better for fairness across classes, but slightly lower overall accuracy because it throws away data to balance things out.

---

## How It's Built

The whole thing runs in a single Colab notebook. The main dependencies are HuggingFace `transformers` and `datasets`, PyTorch, and scikit-learn for evaluation. Each ablation experiment reinitialises the model from scratch so there's no contamination between runs. Everything uses a fixed random seed for reproducibility.

The results get exported as JSON and visualised in the dashboard linked above — scaling curves, quality comparisons, confusion matrices, and per-class breakdowns.

---

## What I Learned

Running ablations yourself teaches you things that reading about them doesn't. Seeing the scaling curve flatten in your own experiment, watching F1 drop because you introduced label noise you put there intentionally — it makes the concepts stick differently. This methodology is what teams at major labs use when deciding how to build their pre-training data mixes, just at a much smaller scale. The principles transfer.
