# Assignment 13 YoloV3 From Scratch on Pascal VOC
#anand gupta
#arghya Mukherjee

<br>

- [try on spaces](https://huggingface.co/spaces/anantgupta129/PyTorch-YoloV3-PascolVOC-GradCAM) 
- [main repo](https://github.com/anantgupta129/TorcHood)
  
# Model Archieture

![](https://miro.medium.com/v2/resize:fit:1200/1*d4Eg17IVJ0L41e7CTWLLSg.png)

# Experiments

## View Trained Model Results

- For a detailed breakdown of the model's performance and visualizations, check the results on [Weights & Biases](https://api.wandb.ai/links/anantgupta129/83aopx49).
  
- Experiment 1: [with mosiac](./notebooks/train_mosiac.ipynb)
    ```
    trainloss: 3.593   | val loss: 3.347

    Class accuracy is: 89.772789%
    No obj accuracy is: 99.000755%
    Obj accuracy is: 74.073151%

    MAP:  0.5215734243392944
    ```
- Experiment 2: [mosiac + mixup](./notebooks/train_mosiac_mixup.ipynb)
    ```
    train loss: 5.241 | val loss: 3.646
    Class accuracy is: 88.029930%
    No obj accuracy is: 99.012306%
    Obj accuracy is: 71.626488%

    MAP:  0.4827963709831238
    ```
