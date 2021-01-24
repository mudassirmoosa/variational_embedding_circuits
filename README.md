# *Variational Embeddings in Quantum Machine Learning*
## **QOSF mentorship program**

**Mentor:** Aroosa Ijaz

**Mentees:** Narges Alavi Samani, Mudassir Moosa, Syed Raza

* Detailed project report in the folder **Report** *

## **Project Description:**

Machine Learning is a potential application for near-term intermediate scale quantum computers with potential for speed-ups over their classical counterparts. Quantum classifiers are quantum circuits that can be trained to classify data in two stages; 1) *Embedding*: the input data is encoded in to quantum states, embedding it to a high-dimensional Hilbert space. 2) *Measurement*: A quantum measurement of the circuit which leads to the output of the model. Usually, the *measurement* part of the circuit is trained but in a recent paper [1] an alternate approach has been adopted where the *embedding* part of the circuit is trained instead. In this work we benchmark various circuit embeddings and cost functions and propose improvements to the state-of-the-art techniques.

## **Key Results:**
1) We benchmark the performance of various variational embedding circuits for classification tasks.
2) We present an alternate to Hilbert-Schmidt cost function, an empirical risk function which can lead to better performance and is illustrated by some toy examples. 
3) In single-wire circuits, the optimization of the Hilbert-Schmidt cost function is a computationally expensive task. We propose a more efficient overlap function that takes a third of the time. 
4) We propose a framework on using Fourier series to quantify the expressivity of the various embedding circuits for classification problems. 

## **References:**

[1] Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, and Nathan Killoran, “Quantum embeddings for machine learning,” arXiv e-prints, arXiv:2001.03622 (2020)
