import classifier.ann.MulticlassDeltaRulePerceptron;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MainTest {
    public final static String FOLDER = "dataset/";
    public final static String TRAINDATASET = FOLDER + "weather.nominal.arff";
    public final static String TESTDATASETJ48 = FOLDER + "test.weather.nominal.arff";

    public static void main(String[] args) throws Exception {
        DataSource datasource = new DataSource(TRAINDATASET);
        Instances trainInstances = datasource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
        Evaluation evaluation = new Evaluation(trainInstances);
        Evaluation wekaEvaluation = new Evaluation(trainInstances);

        MulticlassDeltaRulePerceptron multiclassDeltaRulePerceptron = new MulticlassDeltaRulePerceptron();
        multiclassDeltaRulePerceptron.setLearningRate(0.1);
        multiclassDeltaRulePerceptron.setMomentum(0.05);
        multiclassDeltaRulePerceptron.buildClassifier(trainInstances);
        evaluation.evaluateModel(multiclassDeltaRulePerceptron, trainInstances);
        evaluation.evaluateModel(multiclassDeltaRulePerceptron, trainInstances);

        System.out.println(evaluation.toSummaryString());


    }
}
