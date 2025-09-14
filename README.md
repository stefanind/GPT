This is a complete adaptation of GPT-2 (with some GPT-3 hyperparameters) followed from Andrej Karpathy's YouTube lectures, "Neural Networks: Zero to Hero". Specifically, this is a follow along from the video "Let's reproduce GPT-2." 

The code is mostly his (I take no rights to it), but I rewrote it by following along, added my own comments for educational purposes, and refactored it. Therefore, I understand every piece of code written and the role it plays. 

Currently, I am at the process training. I will be creating an instance through Lambda labs to utilize their resources. Distributed training is set up such that one GPU or multiple can be used. 

Overtime, I will make my own additions to it (and will document them) to see if I can substantially beat GPT-2 benchmarks. 

# Additions:

### Dropout and biases

The first obvious additions that will improve the model is by including dropout regularization and biases in the linear 

### Between shard shuffling

I included between shard shuffling in the data loader because the dataset (fineweb-edu) is collected in parts from specific websites, such as Wikipedia and Arxiv. This collection resulted in the dataset having clear segments of where the data is from. For example, when sharding the data, the first two may only contain info from Wikipedia and the next 3 will be from Arxiv. Hence, when training on the first two shards, the loss will jump up when it switches to the Arxiv shards. So by shuffling the shards, the training loss becomes smoother.
