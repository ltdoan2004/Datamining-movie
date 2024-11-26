import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.trees.M5P;
import weka.classifiers.Evaluation;

import java.util.Random;

public class M5PEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\DELL\\Downloads\\output.csv");
        Instances dataset = source.getDataSet();
        String targetColumnName = "vote_average"; // Replace with your target attribute
        int classIndex = dataset.attribute(targetColumnName).index();
        dataset.setClassIndex(classIndex);

        Classifier m5p = new M5P();

        long startTime = System.currentTimeMillis(); // Start timer
        Evaluation evaluation = evaluateModel(m5p, dataset);
        long endTime = System.currentTimeMillis();   // End timer

        System.out.println("M5P Decision Tree:");
        printEvaluationResults(evaluation);
        System.out.println("Runtime: " + (endTime - startTime) / 1000.0 + " seconds");

    }

    public static Evaluation evaluateModel(Classifier classifier, Instances data) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new Random(1)); // 10-fold cross-validation
        return evaluation;
    }

    public static void printEvaluationResults(Evaluation evaluation) throws Exception {
        System.out.println("MAE: " + evaluation.meanAbsoluteError());
        System.out.println("RMSE: " + evaluation.rootMeanSquaredError());
        System.out.println("Correlation: " + evaluation.correlationCoefficient());
    }
}
