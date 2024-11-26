import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.RandomForest;

import java.util.Random;

public class ModelEvaluation {
    public static void main(String[] args) throws Exception {
        System.out.println("Max Heap Size: " + Runtime.getRuntime().maxMemory() / (1024 * 1024) + " MB");

        // Load the dataset
        DataSource source = new DataSource("C:\\Users\\DELL\\Downloads\\output.csv"); // Path to cleaned dataset
        Instances dataset = source.getDataSet();

        // Set the column "vote_average" as the target
        String targetColumnName = "vote_average"; // Name of the target column
        int classIndex = dataset.attribute(targetColumnName).index(); // Get the index of the target column
        dataset.setClassIndex(classIndex);

        // Initialize classifiers
        Classifier randomForest = new RandomForest();
        Classifier linearRegression = new LinearRegression();

        // Perform 10-fold cross-validation for Random Forest
        System.out.println("Evaluating Random Forest...");
        Evaluation rfEvaluation = evaluateModel(randomForest, dataset);

        // Perform 10-fold cross-validation for Linear Regression
        System.out.println("Evaluating Linear Regression...");
        Evaluation lrEvaluation = evaluateModel(linearRegression, dataset);

        // Print results
        System.out.println("\n--- Results ---");
        System.out.println("Random Forest:");
        printEvaluationResults(rfEvaluation);
        System.out.println("Linear Regression:");
        printEvaluationResults(lrEvaluation);
    }

    // Method to evaluate a classifier using 10-fold cross-validation
    public static Evaluation evaluateModel(Classifier classifier, Instances data) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        long startTime = System.currentTimeMillis(); // Measure runtime
        evaluation.crossValidateModel(classifier, data, 10, new Random(1)); // 10-fold cross-validation
        long endTime = System.currentTimeMillis();
        System.out.println("Runtime: " + (endTime - startTime) / 1000.0 + " seconds");
        return evaluation;
    }

    // Method to print evaluation metrics
    public static void printEvaluationResults(Evaluation evaluation) throws Exception {
        System.out.println("Mean Absolute Error (MAE): " + evaluation.meanAbsoluteError());
        System.out.println("Root Mean Squared Error (RMSE): " + evaluation.rootMeanSquaredError());
        System.out.println("Correlation Coefficient: " + evaluation.correlationCoefficient());
    }
}
