# CAN Signal Inference Using Linear Least Squares

**Course:** ENGG*2910 – Mathematics for Artificial Intelligence  
**Team Members:** [Names]  
**Group Number:** [Group Number]  
**Lab Section:** [Lab Section]

## 1. Problem Definition and Objective

Modern vehicles and machinery use Controller Area Network (CAN) buses to transmit measurements such as engine speed, temperature, pressure, speed, and other sensor values. A CAN frame contains raw binary data, but without a database describing the frame, it is not immediately known which bits represent a particular physical measurement or how the raw number is converted into engineering units.

The objective of this project is to investigate whether linear least-squares regression can recover an unknown CAN signal definition when only two sources of information are available:

1. raw CAN frames; and  
2. synchronized measurements of the physical quantity being transmitted.

The main research question is:

**Can a least-squares-based inference method correctly identify the location and linear scaling of an unknown CAN signal, and how does the number of available observations affect recovery accuracy?**

The investigation will use SAE J1939 CAN messages because J1939 is widely used in trucks, heavy equipment, engines, and other mechanical and mechatronic systems.

For a candidate CAN field, the input is a sequence of extracted raw integer values,

\[
x_1,x_2,\ldots,x_n,
\]

together with synchronized physical measurements,

\[
y_1,y_2,\ldots,y_n.
\]

The desired output is the candidate field that most likely corresponds to the physical measurement, including its CAN message, starting bit, bit length, signedness, byte order, scale factor, and offset.

Success will be measured primarily using Top-1 signal-location accuracy and full-encoding accuracy. The effect of observation count on these measures will also be investigated.

---

## 2. Mathematical Approach and Technical Plan

Many CAN signals use a linear conversion between a raw integer value \(x\) and a physical value \(y\):

\[
y=ax+b,
\]

where \(a\) is the scale factor and \(b\) is the offset.

For \(n\) synchronized observations, this can be expressed as a matrix equation:

\[
\mathbf y=X\boldsymbol{\theta}+\boldsymbol{\epsilon},
\]

where

\[
X=
\begin{bmatrix}
x_1 & 1\\
x_2 & 1\\
\vdots & \vdots\\
x_n & 1
\end{bmatrix},
\qquad
\boldsymbol{\theta}=
\begin{bmatrix}
a\\
b
\end{bmatrix},
\qquad
\mathbf y=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}.
\]

The least-squares estimate is defined by

\[
\hat{\boldsymbol{\theta}}
=
\underset{\boldsymbol{\theta}}{\operatorname{argmin}}
\left\|X\boldsymbol{\theta}-\mathbf y\right\|_2^2.
\]

For each possible CAN field, the program will extract a candidate raw signal \(x\), determine the best values of \(a\) and \(b\), and calculate the residual

\[
\mathbf r=\mathbf y-X\hat{\boldsymbol{\theta}}.
\]

Candidates will then be ranked using measures of fit such as the coefficient of determination \(R^2\), root-mean-square error,

\[
RMSE=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
},
\]

and the proportion of physical observations that can be paired with CAN observations.

The main course concepts used are vectors, matrices, least-squares approximation, projections, residuals, and numerical computation.

Two controlled candidate-search methods will be compared. The baseline will search byte-aligned 8-, 16-, and 32-bit signals. A second search will allow signals to begin at arbitrary bit positions. Comparing the two searches will demonstrate the trade-off between a larger candidate space, computational effort, and signal-recovery performance.

Potential mathematical difficulties include insufficient variation in the raw values, poorly conditioned data, and identifiability. For example, two different candidate fields may produce nearly identical linear relationships over the observed range. Signed and unsigned interpretations may also be indistinguishable if the measured values never cross the sign bit.

---

## 3. Data, Implementation, and Evaluation

A controlled synthetic J1939 dataset will be generated in Python using real J1939 signal definitions as the source of the hidden signal encodings. The inference algorithm will not be given these definitions. The definitions will be retained separately as an answer key and used only after inference for evaluation.

The main experiment will use approximately **25 signals** rather than the complete J1939 database so that the investigation remains manageable. Signals will include common 8-, 16-, and 32-bit representations. Experiments will use approximately

\[
10,\;25,\;50,\;100,\;\text{and}\;200
\]

known observations per signal.

The Python workflow will be:

**J1939 definitions → synthetic physical values → encoded CAN frames → remove signal definitions → candidate extraction → least-squares fitting → candidate ranking → comparison with hidden ground truth**

The inference stage will receive only raw CAN frames and synchronized physical measurements. It will not receive the correct signal locations, factors, offsets, or other database information.

Python will be used for all simulation, inference, and evaluation. NumPy will support vector and matrix calculations, while Matplotlib will be used to produce result plots. The core candidate generation, regression, scoring, and evaluation algorithms will be implemented as part of the project rather than relying on a pre-trained machine-learning model.

Performance will be evaluated using:

- **Top-1 location accuracy:** percentage of signals for which the highest-ranked candidate has the correct CAN location;
- **Top-5 location accuracy:** percentage for which the correct location appears within the five highest-ranked candidates;
- **full-encoding accuracy:** percentage for which location, bit length, byte order, signedness, scale, and offset are all recovered correctly;
- **RMSE and parameter error:** errors in the predicted physical values, scale factor, and offset;
- **runtime and number of candidates tested:** measures of computational effort.

The principal experiment will plot recovery accuracy against the number of observations. A second comparison will examine the accuracy and computational cost of the aligned and arbitrary-bit search spaces.

Because this is primarily a numerical signal-identification investigation rather than a conventional predictive model, correctness will be measured against hidden reference solutions instead of using a conventional classification train/test split.

The expected result is that additional observations will generally make incorrect correlations easier to reject, while expanding the candidate search space will allow more kinds of signals to be recovered at the cost of additional computation.

---

## 4. Responsible Use and Communication

The project will cite the J1939 signal definitions and any external datasets, libraries, or code used in the implementation. Synthetic data will be generated deterministically using recorded random seeds so that experiments can be reproduced.

The inference algorithm and the ground-truth answer key will remain separate. Signal names, correct bit positions, scaling factors, and other information from the source database will not be supplied to the discovery algorithm. This prevents the evaluation data from leaking the correct answer into the inference process.

The limitations of the approach will also be reported. A strong linear correlation does not necessarily prove that two variables have a causal relationship, and some CAN encodings may be mathematically indistinguishable from one another over a limited observation range. Non-linear, enumerated, checksum, and other special-purpose CAN fields are outside the main scope of this investigation.

**AI/LLM acknowledgement:** Large language models, including OpenAI tools, were used to assist with project planning, software development, debugging, documentation, and preparation of written material. All mathematical methods, code, experimental results, and conclusions used in the final project will be reviewed and understood by the team.

## References

1. SAE International, **SAE J1939 Standards Collection**, vehicle-networking specifications for heavy-duty and off-highway applications.
2. NumPy Developers, **NumPy Documentation**, numerical array and linear-algebra tools used by the Python implementation.
3. Matplotlib Development Team, **Matplotlib Documentation**, plotting tools used for experiment visualization.
4. External J1939 database/source files used to construct the controlled synthetic dataset will be cited with their exact version and source in the final submission.