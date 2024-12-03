# Datamining Movie Project

This project demonstrates a comprehensive pipeline for processing and analyzing movie-related data, transforming raw data into meaningful insights using various machine learning models. The repository supports both regression and classification tasks and provides robust model evaluation metrics.

## Prerequisites

Before proceeding, ensure you have the following installed:

-IntelliJ IDEA (Community or Ultimate Edition)

-JDK (Java Development Kit) 8 or higher

-Weka library JAR file (or Maven dependency)

## Steps to Set Up

### 1. **Download the Weka Library**
   
Visit the Weka official website to download the weka.jar file.

Alternatively, if using Maven, add the following dependency to your pom.xml:

     <dependency>
         <groupId>nz.ac.waikato.cms.weka</groupId>
         <artifactId>weka-stable</artifactId>
         <version>3.8.6</version> <!-- Replace with the latest version -->
     </dependency>

### 2. **Add Weka to Your IntelliJ Project**

-Using JAR file:

1. Create a lib folder in your project directory.

2. Place the downloaded weka.jar file inside the lib folder.

3. Add the JAR file to the project:

Right-click your project > Open Module Settings > Libraries > + > Select weka.jar from the lib folder.

-Using Maven:

Add the Weka dependency to your pom.xml as shown in step 1.

### 3. **Install the MTJ (Matrix Toolkit for Java) Library**

-Weka relies on MTJ for matrix operations. Add it to your project:

Using JAR file: Download the MTJ library and add it to the lib folder, just like the Weka JAR.

Using Maven:

      <dependency>
          <groupId>net.sf.s2</groupId>
          <artifactId>mtj</artifactId>
          <version>0.9.14</version>
      </dependency>
      
## Configuring IntelliJ IDEA
### 1. **Increase JVM Heap Memory to 4GB**

Allocate more memory to handle larger datasets and complex models:

Go to Run > Edit Configurations.

Select your application's run configuration.

In the VM Options field, add:
      
      -Xmx4G

2. Fix Module Access Issues
   
Add the following JVM argument to avoid module access errors:

      --add-opens java.base/java.lang=ALL-UNNAMED

Combine this with the heap size argument in the VM Options field:

      -Xmx4G --add-opens java.base/java.lang=ALL-UNNAMED
## Dataset

Filename: TMDB IMDB Movies Dataset.csv

Description: A CSV file containing over 400k records with movie details. It includes information from both TMDb and IMDb databases.

Source: 

      https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset/data

## Overview Data

![Picture2](https://github.com/user-attachments/assets/a787e823-155c-4ef2-8321-73763672f41c)


![Picture4](https://github.com/user-attachments/assets/d23cb6b2-d452-4dbb-b100-911d12d5a82d)


## Classes Overview

-Java Classes

The project contains modular Java classes for evaluating various machine learning models using the Weka API:

1. DecisionStumpEvaluation

Implements the Decision Stump model for baseline performance analysis.

2. GaussianProcessesEvaluation

Evaluates Gaussian Processes for high-accuracy regression tasks.

3. KNNEvaluation

Implements k-Nearest Neighbors (k-NN) for regression or classification tasks.

4. M5PEvaluation

Combines decision trees and linear regression using the M5P algorithm.

5. ModelEvaluation

Generalized model evaluation class supporting metrics for regression and classification.

6. MultilayerPerceptronEvaluation

Implements Multilayer Perceptron (MLP) for capturing non-linear relationships.

7. REPTreeEvaluation

Uses the Reduced Error Pruning Tree (REPTree) model for quick and interpretable decisions.

8. SMOregEvaluation

Implements Support Vector Machines for regression (SMOreg).

9. Main

Entry point for the project, allowing execution and testing of all models.

-Python Scripts

In addition to the Java classes, the project contains Python scripts for data preprocessing, visualization, and comparison of model performance:

1. process_data.py

Handles the preprocessing pipeline for raw data, including cleaning and transformation.

2. download_dataset.py

Downloads and prepares the raw dataset from external sources.

3. convert_to_arff.py

Converts the cleaned data into ARFF format for compatibility with the Weka framework.

4. comparison.py

Compares the runtime and performance metrics of various models.

5. test.py

Contains unit tests for validating the preprocessing and evaluation pipelines.

6. Visualization Folder

visualize_data.py: Visualizes key features of the dataset using plots.

visualize_data_new.py: Enhanced version of the visualization script.

model_evaluation_metrics.png: Image showing evaluation metrics of different models.

![model_evaluation_metrics](https://github.com/user-attachments/assets/b36eb432-1f75-4799-8411-3d02e89d38eb)

model_runtime_comparison.png: Image illustrating runtime comparisons across models.

![model_runtime_comparison](https://github.com/user-attachments/assets/37ddd029-90a7-45b1-bc81-e4a2e761ba57)

## Key Features

-10-Fold Cross-Validation:

Models are evaluated using robust cross-validation techniques to ensure reliable results.

![image](https://github.com/user-attachments/assets/760241ea-1b87-48be-be3d-0cdf71c87a03)

Model Evaluation Metrics:

-Regression:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Correlation Coefficient

-Classification (if applicable):

Accuracy

F1-Score

Confusion Matrix

## Future Improvements

Implement additional feature engineering techniques.

Optimize hyperparameters for each machine learning model.

Expand to include deep learning frameworks like TensorFlow or PyTorch.
