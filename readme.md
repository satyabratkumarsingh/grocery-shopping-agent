### This is a multi agent reinforcement learning application which browses different gorcery stores, 
### extract the product and prices and provides the best recommendation for the user

## The application has 3 parts, the first part extracts data for distilaltion, so that it can train a smaller model to make sense of data 
## The second part is the distillation where we use the data to finetune gemma3 1B model using GRPO
## The final part intrduces multi agent systems using RAY which extracts data from different stores and based on user preferences and provides recommendation. 
## We use PPO here to train model. 


Running Instructions:
1. First, capture the data for distillation, run the cappture_data_for_distillation.py
2. Then Fnetune the Gemma3 (1B) model. rn FineTune_LLM_GRPO.ipynb
3. Save the model and used saved model to run main.py
   
