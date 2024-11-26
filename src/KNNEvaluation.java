import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;

import java.util.Random;

public class KNNEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\DELL\\Downloads\\output.csv");
        Instances dataset = source.getDataSet();
        String targetColumnName = "vote_average";
        int classIndex = dataset.attribute(targetColumnName).index();
        dataset.setClassIndex(classIndex);

        Classifier knn = new IBk(5); // K=5

        // Measure runtime
        long startTime = System.currentTimeMillis();
        Evaluation evaluation = evaluateModelWithProgress(knn, dataset, 10); // 10-fold CV
        long endTime = System.currentTimeMillis();

        System.out.println("K-Nearest Neighbors (K=5):");
        printEvaluationResults(evaluation);
        System.out.println("Runtime: " + (endTime - startTime) + " ms");
    }

    public static Evaluation evaluateModelWithProgress(Classifier classifier, Instances data, int numFolds) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        Random rand = new Random(1); // Seed for reproducibility
        Instances randomizedData = new Instances(data);
        randomizedData.randomize(rand);

        if (randomizedData.classAttribute().isNominal())
            randomizedData.stratify(numFolds);

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
