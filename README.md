# Summer\_Paper\_Reading_2016
eg: 0-ACL16-Ng-Paper_title.pdf

- Abstract
  - Overview:
  - Advantage:
  - Disadvantage:
  - What can I do? / Can I employ its idea?
- Experiments
  - DataSet:
  - Toolkit:
  - Baseline:
  - Result:

[TOC]

### 1-NIPS15-Google-Pointer Network
  [PDF](http://papers.nips.cc/paper/5866-pointer-networks.pdf), 
  [Bib](https://papers.nips.cc/paper/5866-pointer-networks/bibtex)

  - [**Problem-Paper**] the number of target classes depends on the length of input, which is variable.
  - [**Problem-Experiment**] 
  	- finding planar convex hulls
  	- computing Delaunary triangulations
  	- the planar Travelling Salesman Problem
  - [**Model**] instead of using attention to blend hidden units of an encoder to a context vector of a decoder, it
  uses attention as a pointer to select a member of the input sequence as the output.
  - [**Remark**] 
  	- Neural attention model deals with the fundamental problem representing variable length dictionaries, appling to 
  	three distinct algorithmic problems.
	- Idea is fantastic.

　　Authors of this paper propose an attention model based on "Neural machine translation by jointly learning to align 
and translate"[1] by using LSTMs. Their model decides what to predict by using the output of a softmax MLP over the inputs 
as opposed to regular attention model used in [1]\(RNNSearch\), where they used that to use for the convex combination 
of the input annotations to predict where to attend to translate in the target language. 
The output of the softmax-MLP predicts the location of the output at each timestep directly from the input sequence. 

　　The advantage of this approach is that, the output softmax dictionary only depends on the length of the input 
sequence instead of on the number of possible instances in the problem. 
The authors have provided very interesting experimental results on three different discrete optimization problems.

### 2-ACL16-IBM-Addressing Limited Data for Textual Entailment Across Domains
  [PDF](https://arxiv.org/pdf/1606.02638v1.pdf),
  [!Bib](~)
	  	
  - [**Problem-Paper**] exploit unlabled data to improve F-score for TE task. 
  - [**Problem-Experiment**] find all sentences in a corpus that entail a given hypothsis.<br/>
  Domain: Newswire (RTE-5,6,7) & Clinical (self construct)
  - [**Model**] Tradition Features + self-training/activate learning
  - [**Conclusion**] The author analysis the possible explanation for the improvement:
  Class Distribution. down-sampling and up-sampling is not useful. 
  Activate learning will sampling positive examples, thus it can match the performance with fewer examples. 
  - [**Remark**] Experiment is beautiful and convising, Information Retrival Method.

　　This paper illustrates that how self-training will influence the classification and why active learning 
will reduce the examples to only 6 percent. <br/> 
　　The author experiments on two domains -- newswire and clinical. First, the author creates an entailment dataset 
for clinical domain with human annoted. Second, he builds a highly competitive supervised system, called ENT. Last, he
explore two strategies - self-training and active learning to address the lack of labled data. Experiment is done in 
detail and convincing.

### 3-ACL16-Stanford-A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task
  [PDF](https://arxiv.org/abs/1606.02858),
  [!Bib](~),
  [!Github](https://github.com/danqi/rc-cnn-dailymail)

  - [**Problem**] CNN/Daily Mail Reading Comprehension Task
  - [**Model**] 
  	- Traditional Features
  	  - Feature ablation analysis  
  	- Attention Neural Network(followed 5-NIPS15-NYU-End-To-End Memory Networks)
  - [**Related Dataset**]
    - CNN/Daily Mail (2015)
    - MCTest（2013）
    - Children Book Test （2016）
    - bAbI （2016）
  - [**Data Analysis**] breakdown of the Examples, Analysis the perfomance on each categories(although on small dataset).
  - [**Remark**] Also we can construct traditional ML and NN, **data analysis is important**, without this, Experiment
  seems to be inconvincing.

　　This paper conducts a thorougn examination of CNN/Daily Mail Reading Comprehension Task, which origin from the idea 
that a bullet point usually summaries one or several aspects of the article. **If the computer understands the content
of the artticle. It should be able to infer the missing entity in the bullet point.**

　　two supervised systems are implemented -- a conventional entity centric classfier and an end to end neural network.
Expriment shows that the straight-forward NLP system, compared with origin frame-semantic parser[^1], obtain accuracies
of 72.4% and 75.8％ on these two datasetｓ.

　　Besides, the author **extracts 100 examples to analysis the results**. She roughly classify the examples into 6 categories, 
i.e., Exact Match, Paraphrase, Parial clue, Multiple sent, (Coref.Error, Hard), the last two is hard for human to obtain
the correct answer.

[^1]: Teaching Machine to read and comprehend, NIPS15, Hermann et.al

### 4-NAACL16-CMU-Hierarchical Attention Networks for Document Classification
  [PDF](http://aclweb.org/anthology/N/N16/N16-1174.pdf),
  [Bib](http://aclweb.org/anthology/N/N16/N16-1174.bib)

  - [**Goal**] Hierarchical Attention Networks for Document Classification
  - [**Problem**]
  	- a). Sentiment Estimation
	  	- Data Set: Yelp reviews, IMDB reviews, Amazon reviews
  	- b). Topic Classification
  		- Data Set: Yahoo answers
  - [**Model**]
  
  	![Hierarchical Attention Networks](/figs/4a.png)
 
    - (i)  it has a hierarchical structure that mirrors the hierarchical structure of documents; 
    - (ii) it has two levels of attention mechanisms applied at the word and sentence-level, enabling it to attend 
    differentially to more and less important content when constructing the document representation
    - The context vector u_w can be seen as a high level representation of a fixed query “what is the informative word” 
    over the words like that used in memory networks (Sukhbaatar et al., 2015; Kumar et al., 2015).
  - [**Remark**]
    - **Modification of Model**: Hierarchical + Attention

### 5-NIPS15-NYU-End-To-End Memory Networks
  [PDF](http://papers.nips.cc/paper/5846-end-to-end-memory-networks), 
  [Bib](http://papers.nips.cc/paper/5846-end-to-end-memory-networks/bibtex),
  [Theano](https://github.com/npow/MemN2N),
  [Tensorflow](https://github.com/seominjoon/memnn-tensorflow)
  
  - [**Goal**] introdce a **neural network** with a **recurrent attention** over a possibly large **external memory**. 
  - [**Problem-Experiment**] 
    - a). synthetic question answering (Reading Comprehension Task)
    - b). Language Model
  - [**Dataset**]
    - a).bAbI
    - b).Penn Tree Bank & Text8
  - [**Model**] <br/>
    **Notation**:
	- input: sentences: $$x_1, x_2, \ldot, x_i$$, question: q
	- output: answer:a
	- Variable: A, B, C, W
	- $$ shape(x_i) = d; shape(A) = (d, V); shape(m_i) = d; shape(input) = (len(sentences), d) $$
      ![End-To-End Memory Networks](/figs/5a.png)
    - Sentence Representation: BoW & **Position Encoding** (slightly different from BoW, add position information), 
      $$ m_i = \sum_{j}Ax_{ij} $$  
    - Temporal Encoding: QA tasks require some notion of temporal context, $$ m_i = \sum_{j}Ax_{ij} + T_{A}(i)$$ 
    - Random Noise: to reqularize T_A, randomly add 10% of empty memories to the stories.
    - The capacity of memory is restricted to the most recent 50 sentences.
    - **Since the number of sentences and the number of words per sentence varied between problems, a null symbol(all 
    zero) was used to pad them all to a fixed size**
  - [**Remarks**]
    - **How to write a new Model with not the state-of-art performance**?
      - incuction previous model to this model (LSTM, Attention, ...)
      - Compare with other related model (where is the difference?)
      - How can the model changes?
      - What can the model apply?
    - **Related Works** deserves to be learned. 
    - **Code** deserves to be implenment by myself.
   
　　This Paper introduce a neural nerwork with a recurrent attention model over a possibly large external memory.

　　**The memory in RNNs model is the state of the network, which is latent and inherently unstable over long timescales. 
The LSTM-based Model address this through local memory cells which lock in the network from the past. This model differs
from these in that it uses a global memory, with shared read and write functions.**    
    
　　This model also related to attention mechanism in Banhadnau's work[^1], although Banhadnau's work[^1] is only over a
single sentence rather than many.

　　This approach is compettive with Memory Networks, but with less supervision.

Next is quoted from yanranli's blog. The summary deserves me to leaned. you can refer to [here](http://yanran.li/peppypapers/2016/01/09/nips-2015-deep-learning-symposium-part-ii.html) for
more details. 
   > And the authors attempt several ways in this paper to fulfill their goal. 
   > First, the single-layer or multi-layer, and then the transformation of feature space. 
   > If one separate the output of the end-to-end memory networks, they can be parallized with typical RNN. 
   > The output comprises of two parts, internal ouptut and external output, 
   > which can be parallized to RNN’s memory and predicted label, respectively.

[^1]: Neural machine translation by jointly learning to align and translate. ICLR15, Bahdanau, Cho, and Bengio


### 6-NAACL16-Sultan-Bayesian Supervised Domain Adaptation for Short Text Similarity
  [PDF](http://www.aclweb.org/anthology/N/N16/N16-1107.pdf), 
  [Bib](http://www.aclweb.org/anthology/N/N16/N16-1107.bib)

  - [**Problem**] Domain Adaptation for Short Text Similarity
  - [**Model**]  A two-level hierachical Bayesian model -- Each $w_d$ depends not on its domain-specific observations
  (first level) but also on information derived from the global, shared parameter $w*$ (second level). And the 
  hierarchical structure (1) jointly learns global, task-level and domain-level feature weights, (2) retaining the 
  distinction between in-domain and out-of-domain annotations.
  - [**Features**]
    - monolingual word aligner
    - cosine similarity from 400-dimensional embedding(Baroni et.al, 2014)
  - [**Experiment**]
    - a). Short Text Similarity(STS), 10 domains
    - b). Short Answer Scoring(SAS), Dataset: Mohler et al., 2011
    - c). Answer Sentence Ranking(ASR), Dataset: Wang et al., 2007, TREC8-13
  - [**Remarks**]
    - 	Although this is traditional feature method, and the results is not inspiring, the author construt **amount of 
    Analysis** to show the advantage of the system and answer why it does not perform well(**because of the data, smile**). 

	
### 7-NAACL16-Lu Wang-Neural Network-Based Abstract Generation for Opinions and Arguments
  [PDF](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.pdf),
  [Bib](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.bib)
  
  **Excellent work, clear structure, read it more times**

  - [**Problem**] Abstract generation for opinioons and arguments
  - [**Model Step-by-step**]
     - Data Collection, the dataset can be found [here](http://www.ccs.neu.edu/home/luwang): 
       - movie reviews, from www.rottentomatoes.com
       - arguments on contraoversial topics, from idebate.org
     - Step 1: **Problem Formulation**, the ... task is defined to as finding y, which is the most likely sequence of word... such that: formulation
     - Step 2: **Decoder**, LSTM model for long range dependencies.
     - Step 3: **Encoder**, Bi-LSTM + Attention, Attention is userd to know how likely the input word is to be used to generate the next word in summary.
     - Step 4: **Attention Over Multiple Inputs**: It depends on task.
     - Step 5: **Importance Estimation to sub-sampling from the input**: because there are two problems with this approach. Firstly, the model is sensitive to the order of text units (a paragraph); Secondly, time cost too much.
     - Step 6: **Post-processing**: re-rank the n-best summaries; it is directly related to the final goal.
  - [**Experiment:anwser the question from model**]
     - Question 1: How is the performance of component? -- Importance Estimation Evaluation(Step5)
     - Question 2: What is the model performance for automatic summary?
     - Question 3: What is the model performance according to human?
     - Question 4: What is the hyper-parameter K in sub-sampling effect? (Step5)
     - Question 5: Is the post-processing needed? (Step6)
   
　　This work comes from deepmind, it presented a neural approach to generate abstractive summaries for opinionated text.
Attention-based method is employed to find salient inormation from different input text generate an informative and 
concise summary. To cope with the large number of input text, an importance-based sampling mechanism is deployed for 
training.

　　This work applies the attention model to abstract generation. **I think the motivation to build this model is to 
employ attention over different input text (different task may have different question to solve, different model to 
modify, haha)**
   
　　I like this writting skills and research method, such as Bengio, deepmind.


### 8-NIPS15-DeepMind-Teaching Machines to Read and Comprehend
  [PDF](https://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend),
  [Bib](https://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend/bibtex),
  [Tensorflow](https://github.com/carpedm20/attentive-reader-tensorflow)
   
  **What is the guide paper, which will be reseach hot in the future?**

  - [**Problem**]: Large scale training and test datasets have been missing for Reading Comprehension Task (Document, 
  Question) - (Answer).
  - [**Data Collection**]: Summary and paraphrase sentences, with their assosiated documents, can be readily converted 
  to context-query-answer triples using simple entity detection and anonymisation algorithms.
  - [**Models**]
  	- Traditional Methods(Benchmark): Symbolic Matching Models.
  	- Neural Network Models
  	  ![Deep Learning Reader](figs/8a.png) 
      - g(d, q) returns a vector embedding of a document and question pair
      - **The Attentive Reader can be viewed as a generalisation of the application of Memory Networks**, Memory Networks
      employs an attention mechanism at the sentence level, while this at the tokens level but with entire future and past
      context.
      - **Impatient Reader emphasizes on the reread mechanism.** Motivated from " We expect to comprehend deeplier as we 
      reread once more "

      >  Impatient Reader. This model is really interesting and intuitive, which emphasizes on the reread mechanism. 
      >  That is to say, for each token in each query, a whole document is read once through. One token, one document. 
      >  Next token, again this document. That’s what the reread is. I interpret such reread mechanism as a gradual 
      >  comprehension procedure. When a human being is reading a tough article, s/he will read again and again. 
      >  We expect to comprehend deeplier as we reread once more. Such motivation, behind this reread mechanism, if any, 
      >  will make larger impact when predicting beyond a token level output. Therefore, I think this mechanism is 
      >  worthy of implementing in more tasks.
  
  - [**Experiments**]
    - How difficult is this task? Traditional excellet model, **Danqi Chen [^1] achieve great results with Features**
    - Traditional Methods versus Neural Models?
    - Which component contributes to the end performance?

[^1] 3-ACL16-Stanford-A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task

### 9-AAAI16-Mueller-Siamese Recurrent Architectures for Learning Sentence Similarity
  [PDF](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf),
  Bib
 
  **A new top resluts for Relateness on SICK: 0.8822(pearson)**
  
  - [**Model**]
    - replace the top softmax layer with l1 norm similarity function
    - Sentence Representation can capature the following information:
      - negation, not, no
      - topic
      - Entailment classsification 
    - I think the model is to enforcement the similarity of the two sentences, which is direct and efficient.
  - [**Experiments TODO**]
    - replace the top layer
    - learn how to fine-grain the network.
    - Read this paper again.
    - Read the latest related paper from reference.


### 10-NIPS15-Ryan-Skip-Thought Vectors
  [PDF](http://papers.nips.cc/paper/5950-skip-thought-vectors),
  [Bib](http://papers.nips.cc/paper/5950-skip-thought-vectors/bibtex),
  [Review](http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips28/reviews/1826.html)
  [Theono](https://github.com/ryankiros/skip-thoughts)

  **This paper presented an encoder-decoder model to learn a generic, distributed sentence encoder.**
  
  - [**Problem**]: Sentence Representation. Sentence vectors ate extracted and evaluated with linear models on 8 tasks. 
  - [**Motivation**]: 
    > In this paper we abstract away from the composition methods themselves and consider an alternative loss function
    > that can be applied with any composition operator. **(step1 purpose: what we want to do?)** 
    > 
    > We consider the following question: is there a task and a corresponding loss that will allow us to learn highly 
    > generic sentence representations?  **(step2 question: The key problem?)** 
	>
	> We give evidence for this by proposing a model for learning high-quality sentence vectors without a particular
	> supervised task in mind. **(step3 plan1: solution we reject)** 
	> 
	> Using word vector learning as inspiration, we propose an objective function that abstracts the skip-gram model to 
	> the sentence level. That is, instead of using a word to predict its surrounding context, we instead encode a sentence
	> to predict the sentences around it. Thus, any composition operator can be substituted as a sentence encoder and 
	> only the objective function becomes modified.  **(step4 our work: step by step, more and more deep)** 
	> 
	> Figure 1 illustrate the model. We call our model **skip-thoughts** and vectors induced by our model are called
	> **skip-thought vectors**. **(step5 Others: Figure or Name)** 

    ![Skip-Thoughts Model](figs/10a.png)

### 11-EMNLP15-Hua He-Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks
  [PDF](http://aclweb.org/anthology/D/D15/D15-1181.pdf),
  [Bib](http://aclweb.org/anthology/D/D15/D15-1181.bib)

  **Multi-Perspective: 1). Multiple granularities 2). Window size 3). Multiple types of pooling 4). Similarity Measurement: Multiple distance functions**
  
  - [**Remarks**]:
    - This model tries to explore multi-perspective method on CNNs, so how to combine the local component I think is the key problem in the experiment, and the author proposed **building block** as a set of groups, which make the whole process clear and efficient.
  - [**Experiments TODO**]   
    - treat each **Perspective** as **Component** implemented in CNNs.
    - analysis different combination's efficient
    - add the thought of **Semantic Similarity** into the **Model**.

### 12-NIPS13-Mikolov-Distributed Representations ofWords and Phrases and their Compositionality
  [PDF](http://arxiv.org/pdf/1310.4546v1.pdf),
  Bib

  [ICLR13-Mikolov-Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf),
  Bib

  - [**Problem**] the cost of computing the gradient of the Skip-gram model is proportion to the vocabulary size.
  - [**Method**]
   - **Hierachical Softmax**
     - Build a binary Huffman tree as the representation of the output layer with W words as its leaves, and for each non-leaves, explicitly represents the relative probabilities of its chiald nodes.
     - reduce to log(W) 
   - **Negative Sampling**  
     - Assumption: **A good model should be able to differentiate data  from noise by means of logistic regression**.
     - replace the objective of nagative sampling, to distingush the target word from draws from the noise distrubution 
  - [**Others**]
    - **SubSampling of Frequent Words**
      - the most frequent words usually provide less information value than the rare words.
      - discarded with a probability
    - **Phrases Vector**
      - replace words to phrases
      - how to extract phrases? -- words that appear frequently together, and infrequently in other contexts.

　　This two paper proposed the skip-gram model, and tries to solve two aspects problems: 1). How to make it computable? 2). How to make it more semantical? 

　　As to the first problem, the author tries two method, the one is to replace the softmax as hierachical softmax, which reduce the time complexity to log(W), and the other is to replace the objective with negative sampling. 

　　And when it comes to the second problem, the author tries some tricks. Firstly, he subsampling the frequent words since the vector representations of frequent words do not change significantly. Secondly, he treat phrases as a kind of word to train phrase vectoe, because many phrases have a meaning that is not a simple composition of the meaning of the its individual words.

　　**How to build a model? I think this papers pointed out the right directions.**

### 13-ACL16-Yandex-Siamese CBOW: OptimizingWord Embeddings for Sentence Representations
  [PDF](https://arxiv.org/abs/1606.04640),
  Bib,
  [Theono](https://bitbucket.org/TomKenter/siamese-cbow)
  
  **Siamese CBOW trains word embedding directly for the purpose of being averaged to produce better sentence representation.**
  
  **Still, a question need to be answered -- How to select negative samples? random? according to similarity? refer to the code! (Update: Both in word2vec[line 442-446] & Siamese CBOW[line 432-448], random slect negative samples!)**
 
  - [**Model**]
    - Constructing a supervised training criterion by having our network predict sentences occuring next to each other in the training data, which is similar to **Skip-thought**. 
  - [**Related Work**]
    - Word2vec
    - Skip-thought  
  - [**Training Set**]
    - Toronto Book Corpus: 74,004,228 setences; 1,057,070,918 tokens, originating from 7087 unique books.
    - consider tokens appearing 5 times or more, which leads to a vocabulary of 315,643 words.
    - http://www.cs.toronto.edu/~mbweb 
  - [**TODO**]
    - **replace word2vec in STS with this model**
 
### 14-ACL16-Stanford-A Persona-Based Neural Conversation Model
   [PDF](http://arxiv.org/pdf/1603.06155v2.pdf),
   [!Bib]()

   - [**Problem**]: Inconsistent responses via SEQ2SEQ model.
     > *message*: How old are you? <br/>
     > *response*: 16 and you?     <br/>
     > *message*: What's your age? <br/>
     > *response*: 18.
   - [**Model**]
     - Idea: a). Prior Knowledge, b). User Modeling 
       - This model represents each individual speaker as a vector or embedding, which encodes speaker-specific information (e.g., dialect, register, age, gender, personal information) that influences the content and style of her response. 
       - **Speaker Model**: As in standard SEQ2SEQ models, we first encode message S into a vector representation using the source LSTM. Then for each step in the target side, hidden units are obtained by combining the represeantation produced by the target LSTM at the previous time step, the word representations at the current time step, the speaker embedding.
       - **Speaker-Addressee Model**: replace speaker embedding with an interactive representation by combining two speaker embedding.
   - [**Remark**]
     - It is a natural idea to embed our common knowledge or our prior knowledge with a user specific vector representaion and update it throughout the generation process, as human learns from current conversation.
     - **But what do we learn from our daily conversation?**
       - knowledge? thinking is more likely to get it.
       - strength our memory? maybe.
       - interactive? Interaction between two users' knowledge representaion. **How to model?** 

### 15-ACL16-Percy Liang-How Much is 131 Million Dollars Putting Numbers in Perspective with Compositional Descriptions
  [PDF](http://nlp.stanford.edu/pubs/chaganty2016perspectives.pdf),
  [!Bib](~)

  **How to represent number?**
  
  - [**Problem**]: 

### 16-ICLR16-facebook-SEQUENCE LEVEL TRAINING WITH RECURRENT NEURAL NETWORKS



### TD1-ACL16-Microsoft-Deep Reinforcement Learning with a Natural Language Action Space
  [PDF](http://arxiv.org/pdf/1511.04636v5.pdf),
  [!Bib](~)
 

# Other Papers:

### 1-ACL16-Simple PPDB: A Paraphrase Database for Simplification
  [PDF](http://cis.upenn.edu/~ccb/publications/simple-ppdb.pdf),
  [!Bib](~)

### 2-harvard-Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks
  [PDF](https://arxiv.org/abs/1606.07461),
  [code](http://lstm.seas.harvard.edu/)
  
  LSTM Visual Analysis

## NAACL
<del> A Neural Network-Based Abstract Generation Method to Opinion Summarization

<del> Bayesian Supervised Domain Adaptation for Short Text Similarity

Clustering Paraphrases by Word Sense

Convolutional Neural Networks vs. Convolution Kernels: Feature Engineering for Question Answering

DAG-structured Recurent Neural Networks for Semantic Compositionality

Deep LSTM based Feature Mapping for Query Classification

Dependency Based Embeddings for Sentence Classification Tasks

Dependency Sensitive Convolutional Neural Networks for Modeling Sentences and Documents

<del> Hierarchical Attention Networks for Document Classification

Learning Distributed Representations of Sentences from Unlabelled Data

Pairwise Word Interaction Modeling with Neural Networks for Semantic Similarity Measurement

Multi-way, Multilingual Neural Machine Translation with a Shared Attention Mechanism    
