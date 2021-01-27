# **Variational Embeddings in Quantum Machine Learning**
## **[QOSF mentorship program](https://qosf.org)** 
### January 23rd, 2021

**Mentor:** [Aroosa Ijaz](https://aroosaijaz.github.io/)

**Mentees:** [Narges Alavi Samani](https://www.linkedin.com/in/narges-alavi-samani/), [Mudassir Moosa](https://www.linkedin.com/in/mudassir-moosa/), [Syed Raza](https://www.linkedin.com/in/syedraza22/)

The project [report](https://github.com/mudassirmoosa/variational_embedding_circuits/tree/master/Report) in the folder `Report'.

## **Project Description:**

Machine Learning is a potential application for near-term intermediate scale quantum computers with possible speed-ups over their classical counterparts. Quantum classifiers are quantum circuits that can be trained to classify data in two stages; 1) *Embedding*: the input data is encoded into quantum states, embedding it to a high-dimensional Hilbert space. 2) *Measurement*: A quantum measurement of the circuit to discriminate between classes. Usually, the *measurement* part of the circuit is trained but a recent work [1] adopts an alternate approach where the *embedding* part of the circuit is trained instead, freeing up more precious resources. In this work, we benchmark various embeddings and cost functions and propose improvements. Some key results: 1) We compare the performance of various variational embedding circuits for classification tasks. 2) We present an alternate to Hilbert-Schmidt cost function, an *empirical risk function* which can lead to better performance as illustrated by some toy examples. 3) In single-wire circuits, the optimization of the Hilbert-Schmidt cost function is a computationally expensive task. We propose a more efficient *overlap function* that takes a third of the time. 4) We conjecture a framework to quantify the expressivity of various embedding circuits for classification tasks. 

## **Key Results:**
1) We benchmark the performance of various variational embedding circuits for classification tasks.
2) We present an alternate to Hilbert-Schmidt cost function, an empirical risk function which can lead to better performance and is illustrated by some toy examples. 
3) In single-wire circuits, the optimization of the Hilbert-Schmidt cost function is a computationally expensive task. We propose a more efficient overlap function that takes a third of the time. 
4) We propose a framework on using Fourier series to quantify the expressivity of the various embedding circuits for classification problems. 

## **Code Repo Descriptions:**
1) Folder overlap_vs_HS_cost has code for the comparision between the optimization of the Hilbert-Schmidt cost function and the optimization of the overlap. These codes are used in the analysis presented in Sec. (III) of the report.
2) Folder random_embedding_circuits has code for the random variational embedding circuits that we have presented in Sec. (IV) of the report. These codes are used to generate the plots shown in Fig. (3) and Fig. (4) of the report.
3) Folder risk_function/2d_data has code for the analysis of the risk function presented in Sec. (II) of the report. There is also a code for generating a data set shown in Fig. (1) of the report. 
4) Folder Fourier_analysis/1d-QAOA-Fourier has code for performing a Pauli decomposition of the output of a single-wire embedded circuit. See, for e.g., Eqs. (2.3) and (2.4) of the report. 
5) Folder Simulation_of_Variational_Circuits has code for implementing and comparing different variational circuit structures, varying in types of gates, circuit depth, number of qubits, as an embedding circuit for data classification problem.

## **References:**

[1] Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, and Nathan Killoran, “Quantum embeddings for machine learning,” arXiv e-prints, arXiv:2001.03622 (2020)
