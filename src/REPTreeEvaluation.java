import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.classifiers.Evaluation;

import java.util.Random;

public class REPTreeEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\DELL\\Downloads\\output.csv");
        Instances dataset = source.getDataSet();
//        dataset = new Instances(dataset, 0, 10000); // Use the first 10,000 instances
        String targetColumnName = "vote_average";
        int classIndex = dataset.attribute(targetColumnName).index();
        dataset.setClassIndex(classIndex);

        Runtime runtime = Runtime.getRuntime();
        System.out.println("Max Heap Size: " + (runtime.maxMemory() / 1024 / 1024) + " MB");
        System.out.println("Total Heap Size: " + (runtime.totalMemory() / 1024 / 1024) + " MB");
        System.out.println("Free Heap Size: " + (runtime.freeMemory() / 1024 / 1024) + " MB");

        Classifier repTree = new REPTree();

        // Measure runtime
        long startTime = System.currentTimeMillis();
        Evaluation evaluation = evaluateModelWithProgress(repTree, dataset, 10); // 10-fold CV
        long endTime = System.currentTimeMillis();

        System.out.println("REPTree:");
        printEvaluationResults(evaluation);
        System.out.println("Runtime: " + (endTime - startTime) + " ms");
    }

    public static Evaluation evaluateModelWithProgress(Classifier classifier, Instances data, int numFolds) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        Random rand = new Random(1); // Seed for reproducibility
        Instances randomizedData = new Instances(data);
        randomizedData.randomize(rand);

        if (randomizedData.classAttribute().isNominal()) {
            randomizedData.stratify(numFolds);
        }

        for (int i = 0; i < numFolds; i++) {
            Instances train = randomizedData.trainCV(numFolds, i, rand);
            Instances test = randomizedData.testCV(numFolds, i);

            // Build and evaluate the classifier
            classifier.buildClassifier(train);
            evaluation.evaluateModel(classifier, test);

            // Display progress
            int progress = (int) (((double) (i + 1) / numFolds) * 100);
            System.out.println("Progress: " + progress + "%");
        }
        return evaluation;
    }

    public static void printEvaluationResults(Evaluation evaluation) throws Exception {
        System.out.println("MAE: " + evaluation.meanAbsoluteError());
        System.out.println("RMSE: " + evaluation.rootMeanSquaredError());
        System.out.println("Correlation: " + evaluation.correlationCoefficient());
    }
}
