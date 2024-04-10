# HW to Chapter 20 “LM, LSTM, and GRU”

## Non-programming Assignment

### Q1. How does language model (LM) work?

### Answer:        

A Language Model (LM) works by predicting the likelihood of a sequence of words in a language, essentially learning the structure and patterns of the language. The goal is to estimate the probability distribution of various linguistic units, such as words or sentences, enabling the model to predict the next word or sequence of words given a piece of text. Here’s how it operates:

1. Understanding Context:
The LM takes into account the preceding words (context) to predict the next word in a sequence. This context helps the model grasp the flow of language and make informed predictions.
2. Probability Estimation:
It estimates the probability of each word in the vocabulary being the next word in the sequence. This estimation is based on the training data the model has seen during its learning phase.
3. Training on Corpus:
LMs are trained on large corpora of text data. The training involves adjusting the model parameters to minimize the difference between the predicted probabilities and the actual occurrences of words in the training data. This process typically uses techniques such as maximum likelihood estimation.
4. Sequence Modeling:
Modern LMs, especially those based on neural networks like RNNs, LSTMs, and GRUs, can capture long-term dependencies and complex patterns by maintaining an internal state that gets updated as new words are processed. This allows them to remember information about previous words over long sequences, which is crucial for understanding context and generating coherent text.

#### Applications:         

Language models are foundational in various Natural Language Processing (NLP) applications, including:

1. Text Generation:             
Generating coherent and contextually relevant text sequences.
2. Speech Recognition:               
Converting spoken language into text by predicting likely word sequences.
3. Machine Translation:              
Translating text from one language to another by understanding the probability distribution of words in both languages.
4. Autocomplete and Predictive Typing:                 
Suggesting the next word or completing the current word as users type.                

Overall, the effectiveness of an LM is determined by its ability to understand and predict the structure of language, enabling it to perform tasks that require a deep understanding of linguistic patterns and context.

### Q2. How does word prediction work?

### Answer:

### Q3. How to train an LM?

### Answer:

### Q4. Describe the problem and the nature of vanishing and exploding gradients

### Answer:

### Q5. What is LSTM and the main idea behind it?

### Answer:

### Q6. What is GRU?

### Answer: