import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;

import java.util.Random;

public class MultilayerPerceptronEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("C:\\Users\\DELL\\Downloads\\output.csv");
        Instances dataset = source.getDataSet();
        String targetColumnName = "vote_average";
        int classIndex = dataset.attribute(targetColumnName).index();
        dataset.setClassIndex(classIndex);

        Classifier mlp = new MultilayerPerceptron();

        long startTime = System.currentTimeMillis();
        Evaluation evaluation = evaluateModel(mlp, dataset);
        long endTime = System.currentTimeMillis();
        System.out.println("Runtime: " + (endTime - startTime) + " ms");


        System.out.println("Multilayer Perceptron:");
        printEvaluationResults(evaluation);
    }

    public static Evaluation evaluateModel(Classifier classifier, Instances data) throws Exception {
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(classifier, data, 10, new Random(1));
        return evaluation;
    }

    public static void printEvaluationResults(Evaluation evaluation) throws Exception {
        System.out.println("MAE: " + evaluation.meanAbsoluteError());
        System.out.println("RMSE: " + evaluation.rootMeanSquaredError());
        System.out.println("Correlation: " + evaluation.correlationCoefficient());
    }
}
