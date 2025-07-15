、

<div align="center">
  <h1>The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models</h1>
  <br />
  <span style="color:red">📢 <strong><i>If you are interested in our work, please star ⭐ our project.</i></strong></span>

  <h4>
    <a href="https://arxiv.org/abs/2410.16672"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:SPIN&color=red&logo=arxiv"></a>
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=ChnQ.SPIN" />
  </h4>
</div>


## 🌈 Introduction


![Overview Diagram](assets/fig1.png)

Ensuring awareness of fairness and privacy in Large Language Models (LLMs) is critical. Interestingly, we discover a counter-intuitive trade-off phenomenon that enhancing an LLM's privacy awareness through Supervised Fine-Tuning (SFT) methods significantly decreases its fairness awareness with thousands of samples. To address this issue, inspired by the information theory, we introduce a training-free method to **S**uppress the **P**rivacy and fa**I**rness coupled **N**eurons (**SPIN**), which theoretically and empirically decrease the mutual information between fairness and privacy awareness. 

- Extensive experimental results demonstrate that **SPIN eliminates the trade-off phenomenon and significantly improves LLMs' fairness and privacy awareness simultaneously without compromising general capabilities**, e.g., improving Qwen-2-7B-Instruct's fairness awareness by 12.2\% and privacy awareness by 14.0\%. 
- More crucially, **SPIN remains robust and effective with limited annotated data or even when only malicious fine-tuning data is available**, whereas SFT methods may fail to perform properly in such scenarios. 
- Furthermore, we show that **SPIN could generalize to other potential trade-off dimensions**. 

We hope this study provides valuable insights into concurrently addressing fairness and privacy concerns in LLMs and can be integrated into comprehensive frameworks to develop more ethical and responsible AI systems.



##  🚩Main Results

<!-- <p style="color: #0288D1; font-size: 18px; font-weight: bold;">
    SPIN enhances LLM’s awareness of fairness and privacy simultaneously without compromising general capabilities.
</p> -->

**SPIN enhances LLM’s awareness of fairness and privacy simultaneously without compromising general capabilities.**

<img src="./assets/table1.png" width="700" alt="Table 1">

<img src="./assets/table3.png" width="700" alt="Table 3">

<!-- ![Table 1](assets/table1.png) -->

<!-- ![Table 3](assets/table3.png) -->

---

<!-- <p style="color: #0288D1; font-size: 18px; font-weight: bold;">
    SPIN remains robust even when only malicious fine-tuning data is available.
</p> -->

**SPIN remains robust even when only malicious fine-tuning data is available.**

<img src="./assets/fig3.png" width="700" alt="Figure 3">
<!-- ![Figure 3](assets/fig3.png) -->

---

<!-- <p style="color: #0288D1; font-size: 18px; font-weight: bold;">
    SPIN encourages the model to produce more cautionary language related to fairness and privacy.
</p> -->

**SPIN encourages the model to produce more cautionary language related to fairness and privacy.**

<img src="./assets/fig5.png" width="700" alt="Figure 5">

<!-- ![Figure 5](assets/fig5.png) -->


## 🚀Quick Start


### 🔧Requirements

The following pakages are required to run the code:

- python==3.11.5

- pytorch==2.1.2

- transformers==4.40.0

- datasets==2.18.0

### 🌟Usage

**1. Compute and save the importance score**

```bash
cd src/

datasets=(
    "beaver_train330k_privacy_safe_1k"
    "beaver_train330k_fairness_safe_1k"
    "alpaca_cleaned_no_safety"
)

for dataset in "${datasets[@]}"; do
    python compute_importance_score.py \
    --model your_model \
    --model_path your_model_path \
    --nsamples 128 \
    --dataset $dataset
done
```

**2. Run and evaluate SPIN**

```bash
cd src/

python main.py \
--model your_model \
--model_path your_model_path \
--nsamples 128 \
--dataset1 beaver_train330k_privacy_safe_1k \
--dataset2 beaver_train330k_fairness_safe_1k \
--target_module mlp \
--p 5e-7 \
--q 5e-7
```


## 📝License
Distributed under the Apache-2.0 License. See LICENSE for more information.


## Acknowledgements
Some code in this project is adapted from resources provided by the following repositories: 
- https://github.com/boyiwei/alignment-attribution-code
- https://github.com/OpenSafetyLab/SALAD-BENCH

We greatly appreciate the contributions of the original authors.


## 📖BibTeX
```
@article{qian2024tug,
  title={The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models},
  author={Qian, Chen and Liu, Dongrui and Zhang, Jie and Liu, Yong and Shao, Jing},
  journal={arXiv preprint arXiv:2410.16672},
  year={2024}
}