# LLM Training Labs

Python labs on how large language models are trained and used, built step by step from the underlying probability and loss calculations. I developed these labs for the discussion sections of STATS 425 (*Large Language Models in Text Mining*) at UCLA in Fall 2026.

Each lab builds on the previous one: students compute quantities by hand, predict how a computational choice will change model behavior, run the code to test the prediction, and explain what the results support. Every notebook runs in Google Colab on a standard CPU runtime; no GPU or account setup is required.

## Labs

| Week | Lab | Topics | |
|---|---|---|---|
| 1 | [Loss and perplexity](Week_1_LLM_Lab.ipynb) | Tokenization, next-token probabilities, log loss, and perplexity for a pretrained model | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_1_LLM_Lab.ipynb) |
| 2 | [Supervised fine-tuning](Week_2_SFT_Lab.ipynb) | Pretraining versus fine-tuning, prompt–response examples, response-only loss, gradient updates, and evaluation on held-out examples | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_2_SFT_Lab.ipynb) |
| 3 | [Text generation](Week_3_Text_Generation_Lab.ipynb) | Token-by-token generation, greedy decoding, sampling and temperature, and top-k and top-p sampling | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_3_Text_Generation_Lab.ipynb) |
| 4 | [Reinforcement learning with verifiable rewards](Week_4_RLVR_Lab.ipynb) | Verifiers and rewards, sampling from a policy, the REINFORCE policy-gradient update, and testing the trained policy | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_4_RLVR_Lab.ipynb) |
| 5 | Reinforcement learning with calibration rewards | Rewards for reporting confidence, and how they shape a model's probabilities | Coming soon |

Weeks 1–3 use the pretrained [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) model. Week 4 uses a small model whose full calculation can be inspected, then connects the same update to language models.

## Running the labs

**In Colab:** click the badge next to a lab and run the cells in order.

**Locally:** install Python 3.10+ with `torch` and `transformers`, then open the notebooks in Jupyter.

```bash
pip install torch transformers jupyter
jupyter notebook
```

Each lab ends with review questions and answers that can be opened after discussion.

## Author

[Ali Kaazempur-Mofrad](https://alimofrad.com), Department of Statistics and Data Science, UCLA.
