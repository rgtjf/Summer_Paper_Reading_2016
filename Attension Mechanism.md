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