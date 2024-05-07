# nanogpt

This is a simple implementation of decoder only transformer from [https://arxiv.org/abs/1706.03762](Attention is all you need) paper. It doesnot include any cross attention components from the paper. I have implemented a simple self attention with multi head attention. 

As for the tokenizer, I have implemented a simple character level tokenizer, that encode each character from the training data into embedding space and back.

# Training
The model was trained on a Friends screenplay. The data contained about 1M characters. The model was trained on my RTX 2060 GPU for about 1 hour (5000 iteractions) and validation after which started to taper at cross validation loss of ~1.4. You can find the results in the below screenshot. The language generated is not close to the real english, but neverthless the structure resembles a screenplay with all character names correctly spelt. Given that it is a very small model, trained on a tiny data with miniscule compute power the results are quite impressive.

# Extending it beyond language tasks
I tried training it on some time series data and validating the predictions generated from the model. I have changed the tokenizer from character level tokenizer to a bin based tokenizer (If you think about it, character level tokenizer will not work for forecasting task as the data will be too sparse and we will not be capturing the entire context of the numbers). In binned tokenizer, i have scaled the data using StandardScaler and divided the data into 100 bins, with each bin containing 1% of the data. This way of tokenization will introduce some error when decoding back from bins, but for now its okay!! I have trained the model on some dummy data that i generated.

The results were not impressive, it was able to predict cyclic seasonalities to a Ok level, but i think the model just memorized the training data and its repeating. As for the linearly increasing data, the model was unable to generate a continously increasing forecasts. I assume that the model didnot have enough training data to learn from. There are multiple papers which have successfully trained LLMs for few shot and zero shot forecasting and few of them have trained the LLMs on a large forecasting data with multiple kinds of timeseries. I will explore this approach or any other interesting approached in future(may be) and extend this tiny model to forecasting usecase.
