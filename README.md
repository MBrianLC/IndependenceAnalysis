# IndependenceAnalysis

Independence analysis on some of the main batteries of statistical tests of randomness.

For each battery, we display the following results:
- Pearson correlation matrix
- Mutual information matrix
- Matrices resulting from applying a Kolmogorov-Smirnov test both to the p-values from the Pearson correlation, and the p-values from the mutual information (100 p-values each)
- Matrices of means of the previous p-values

We have carried out the analyses on the p-values obtained in each battery (execution on a set of 1,000,000 sequences). In some batteries, we have also performed the analyzes on statistics.

We also include the code used (in Python) to obtain these calculations so that anyone can use it in future analyzes.

Some of these results are shown in the article "A new approach to the independence of statistical tests of randomness" written by Marcos Brian Leiva Cerna, Elena Salomé Almaraz Luengo, Luis Javier García Villalba, and Julio Hernandez-Castro.
