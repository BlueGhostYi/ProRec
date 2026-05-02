# ProRec
**LLM-derived Profiling for Recommendation (ProRec)**

ProRec is an integrated library that brings together our latest advances in LLM-based user profiling for recommendation. 

In recent years, pre-trained Large Language Models (LLMs) have demonstrated remarkable generalization ability for recommender systems. The first line of work involves fine-tuning LLMs to serve as holistic recommendation models; however, this approach incurs high computational costs and often suffers from poor generalization across diverse recommendation scenarios. An alternative strategy treats LLMs as plug-and-play enhancement modules, improving base recommender models by incorporating rich semantic priors. As illustrated in the figure (a), to exploit their advantages in text understanding and generation, LLMs are used to encode user interactions and auxiliary information into compact representations (e.g., **user or item profiles**). 

<p align="center">
<img src="profiling.png" alt="profiling" />
</p>

As illustrated in the figure (b), the objective of user profiling is to leverage LLMs to construct high-quality user profiles that capture user preferences for downstream recommendation tasks. However, when user–item interactions are highly sparse or exhibit complex patterns, one-shot profiling often produces unreliable or even meaningless results (as shown in figure (c)). Such profiles may misrepresent user interests and introduce noise during training, thereby hindering subsequent recommendation performance.

Based on this, **ProRec** builds upon prior research works such as KAR and RLMRec to conduct a more fine-grained investigation of the aforementioned issues.
Specifically, ProRec is an integrated recommendation model library that encompasses our latest research advances in LLM-based user profiling for recommendation. It covers how to construct user profiles, how to align them with different types of recommendation models, and how to maximize the utilization of their semantic information. Currently, ProRec provides three recommendation models: **DMRec (SIGIR'25)**, **ProEx (KDD'26)**, and **ProMax (SIGIR'26)**. ProRec is still under active development and undergoing continuous refinement. Please stay tuned.

## Environment (based on our test platform)
```
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
scipy == 1.10.1
numpy == 1.24.3
tdqm == 4.65.0
```
> For some special models, additional third-party libraries may be required.

## Examples to Run 
Steps to run the code (MODEL_NAME is the name of the model):
1. In the folder . /configure to configure the MODEL_NAME.txt file;
2. Run main.py `python main.py` and select the identifier of MODEL_NAME or specify through the command line:`python main.py --model=MODEL_NAME`
