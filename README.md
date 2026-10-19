# LLM Training Labs

[Ali Kaazempur-Mofrad](https://alimofrad.com), Department of Statistics and Data Science, UCLA.

Python labs on how large language models are trained and used, built step by step from the underlying probability and loss calculations. I developed these labs for the discussion sections of STATS 425 (*Large Language Models in Text Mining*) at UCLA in Fall 2026.

Every notebook runs in Google Colab on a standard CPU runtime; no GPU or account setup is required.

## Labs

| Week | Lab | Topics | |
|---|---|---|---|
| 1 | [Loss and perplexity](Week_1_Next_Token_Lab.ipynb) | Tokenization, next-token probabilities, log loss, and perplexity for a pretrained model | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_1_Next_Token_Lab.ipynb) |
| 2 | [Supervised fine-tuning](Week_2_SFT_Lab.ipynb) | Pretraining versus fine-tuning, prompt–response examples, response-only loss, gradient updates, and evaluation on held-out examples | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_2_SFT_Lab.ipynb) |
| 3 | [Text generation](Week_3_Text_Generation_Lab.ipynb) | Token-by-token generation, greedy decoding, sampling and temperature, and top-k and top-p sampling | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_3_Text_Generation_Lab.ipynb) |
| 4 | [Reinforcement learning with verifiable rewards](Week_4_RLVR_Lab.ipynb) | Verifiers and rewards, sampling from a policy, the REINFORCE policy-gradient update, and testing the trained policy | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_4_RLVR_Lab.ipynb) |
| 5 | [Baselines and group-relative advantages](Week_5_GRPO_Lab.ipynb) | Responses with several tokens, reward baselines and gradient variance, group-relative advantages (GRPO), and a KL penalty to a reference policy | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_5_GRPO_Lab.ipynb) |
| 6 | [Reinforcement learning with calibration rewards](Week_6_RLCR_Lab.ipynb) | Calibration and reliability diagrams, proper scoring rules, and the RLCR reward for an answer and a stated confidence | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_6_RLCR_Lab.ipynb) |
| 7 | [Learning from preferences](Week_7_Preferences_Lab.ipynb) | The Bradley–Terry model, reward models, the KL-regularized RLHF objective, and direct preference optimization (DPO) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_7_Preferences_Lab.ipynb) |
| 8 | [Uncertainty without retraining](Week_8_Uncertainty_Lab.ipynb) | Agreement among sampled answers, temperature scaling, conformal prediction sets, and answering or abstaining with error control | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_8_Uncertainty_Lab.ipynb) |
| 9 | [Embeddings and retrieval](Week_9_Retrieval_Lab.ipynb) | TF-IDF, cosine similarity, latent semantic analysis, clustering, and scoring answers with retrieved evidence | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_9_Retrieval_Lab.ipynb) |
| 10 | [Putting it all together](Week_10_Synthesis_Lab.ipynb) | One task from start to finish: supervised training, rewards with a KL penalty, calibrated confidence, and answers with a coverage guarantee | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amofrad/llm-training-labs/blob/main/Week_10_Synthesis_Lab.ipynb) |

Weeks 1–3 and the last section of Week 9 use the pretrained [DistilGPT2](https://huggingface.co/distilbert/distilgpt2) model. The other labs use small models whose full calculations can be inspected, then connect each method to language models.

## Running the labs

**In Colab:** click the badge next to a lab and run the cells in order.

**Locally:** install Python 3.10+ with `torch`, `transformers`, `scikit-learn`, and `matplotlib`, then open the notebooks in Jupyter.

```bash
pip install torch transformers scikit-learn matplotlib jupyter
jupyter notebook
```
