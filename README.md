# Machine Learning Mathematics

This repository contains a collection of Jupyter notebooks that cover fundamental mathematical concepts essential for machine learning. Each notebook focuses on a specific topic, providing explanations, formulas, and examples to aid understanding.

## Contents

1. **Plotting a System of Linear Equations**  
   Visualizing solutions to linear equations by plotting them in a coordinate system.

2. **Scalars**  
   Introduction to scalars, the simplest form of data in mathematics, representing single numerical values.

3. **Vectors**  
   Understanding vectors, which are ordered lists of numbers, and their role in machine learning.

4. **Norms and Unit Vectors**  
   Exploring vector norms (measures of vector length) and unit vectors (vectors with a norm of one).

5. **Orthogonal Vectors**  
   Discussing orthogonal vectors, which are perpendicular to each other, and their significance.

6. **Matrices**  
   Introduction to matrices, two-dimensional arrays of numbers, and their operations.

7. **Tensor Transposition**  
   Understanding the transposition of tensors, including matrices, which involves swapping their dimensions.

8. **Basic Tensor Arithmetic**  
   Performing arithmetic operations on tensors, the generalization of vectors and matrices.

9. **Matrix Reduction**  
   Techniques for simplifying matrices, such as row reduction, to solve linear systems.

10. **Dot Product**  
    Calculating the dot product, a fundamental operation that combines two vectors into a scalar.

11. **Frobenius Norm**  
    Measuring the size of a matrix using the Frobenius norm, which generalizes the Euclidean norm to matrices.

12. **Matrix Multiplication**  
    Understanding how to multiply matrices and the applications of this operation.

13. **Matrix Inversion**  
    Finding the inverse of a matrix, which is crucial for solving linear systems.

14. **Affine Transformations**  
    Exploring affine transformations, which combine linear transformations and translations.

15. **Eigenvalues and Eigenvectors**  
    Understanding eigenvalues and eigenvectors, which reveal important properties of matrices.

16. **Matrix Determinants**  
    Calculating determinants, scalar values that provide information about a matrix's properties.

17. **Determinants and Eigenvalues**  
    Investigating the relationship between determinants and eigenvalues in matrix analysis.

## Formulas

Throughout the notebooks, several key formulas are presented. Here are some of the fundamental ones:

- **Dot Product**  
  The dot product of two vectors **a** and **b** is calculated as:

  \[
  \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
  \]

- **Frobenius Norm**  
  The Frobenius norm of a matrix **A** is defined as:

  \[
  \|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2}
  \]

- **Matrix Multiplication**  
  The product of matrices **A** (of size *m*×*n*) and **B** (of size *n*×*p*) results in matrix **C** (of size *m*×*p*), where each element is computed as:

  \[
  c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
  \]

- **Matrix Inversion**  
  The inverse of a matrix **A** is denoted as **A⁻¹** and satisfies:

  \[
  A \cdot A^{-1} = I
  \]

  where **I** is the identity matrix.

- **Eigenvalue Equation**  
  For a square matrix **A**, a scalar λ is an eigenvalue, and a non-zero vector **v** is the corresponding eigenvector if:

  \[
  A \mathbf{v} = \lambda \mathbf{v}
  \]

- **Determinant of a 2×2 Matrix**  
  For matrix **A**:

  \[
  A = \begin{bmatrix}
  a & b \\
  c & d
  \end{bmatrix}
  \]

  the determinant is calculated as:

  \[
  \text{det}(A) = ad - bc
  \]

For detailed explanations and additional formulas, please refer to the individual notebooks in this repository.

## Usage

To explore these concepts:

1. Clone the repository:

   ```bash
   git clone https://github.com/as4401s/Machine-Learning-Mathematics.git
