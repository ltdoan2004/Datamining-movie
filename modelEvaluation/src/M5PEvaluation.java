import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.trees.M5P;
import weka.classifiers.Evaluation;

import java.util.Random;

public class M5PEvaluation {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\DELL\\Downloads\\output.csv");
        Instances dataset = source.getDataSet();

        // Set the target attribute (class index) for the model
        String targetColumnName = "vote_average"; // Replace with your target attribute
        int classIndex = dataset.attribute(targetColumnName).index();
        dataset.setClassIndex(classIndex);

        // Initialize the M5P model (regression tree)
        Classifier m5p = new M5P();

        // Measure runtime
        long startTime = System.currentTimeMillis(); // Start timer
        Evaluation evaluation = evaluateModelWithProgress(m5p, dataset, 10); // 10-fold cross-validation
        long endTime = System.currentTimeMillis();   // End timer

        // Print the results
        System.out.println("M5P Decision Tree:");
        printEvaluationResults(evaluation);
        System.out.println("Runtime: " + (endTime - startTime) + " ms");
    }

    // Method for evaluating the model with progress display during 10-fold cross-validation
    public static Evaluation evaluateModelWithProgress(Classifier classifier, Instances data, int numFolds) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        Random rand = new Random(1); // Seed for reproducibility
        Instances randomizedData = new Instances(data);
        randomizedData.randomize(rand);

        // Stratify if the class attribute is nominal
        if (randomizedData.classAttribute().isNominal()) {
            randomizedData.stratify(numFolds);
        }

        // Perform 10-fold cross-validation
        for (int i = 0; i < numFolds; i++) {
            Instances train = randomizedData.trainCV(numFolds, i, rand);
            Instances test = randomizedData.testCV(numFolds, i);

            // Build and evaluate the model
            classifier.buildClassifier(train);
            evaluation.evaluateModel(classifier, test);

            // Display progress
            int progress = (int) (((double) (i + 1) / numFolds) * 100);
            System.out.println("Progress: " + progress + "%");
        }
        return evaluation;
    }

    // Method to print the evaluation results
    public static void printEvaluationResults(Evaluation evaluation) throws Exception {
        System.out.println("MAE: " + evaluation.meanAbsoluteError());
        System.out.println("RMSE: " + evaluation.rootMeanSquaredError());
        System.out.println("Correlation: " + evaluation.correlationCoefficient());
    }
}
