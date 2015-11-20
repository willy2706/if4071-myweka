/**
 * Created by nim_13512065 on 9/24/15.
 */


import classifier.ann.DeltaBatchRule;
import classifier.ann.DeltaRulePerceptron;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public final static String FOLDER = "myweka/dataset/";
    public final static String TRAINDATASET = FOLDER + "masayu.dataset1.arff";
    public final static String TESTDATASETJ48 = FOLDER + "test.weather.nominal.arff";
    public static void main(String[] args) throws Exception {
        DataSource datasource = new DataSource(TRAINDATASET);
        Instances trainInstances = datasource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        Evaluation evaluation = new Evaluation(trainInstances);
        Evaluation wekaEvaluation = new Evaluation(trainInstances);


        /*what we will use*/
//        DeltaBatchRule deltaBatchRule = new DeltaBatchRule();
//        deltaBatchRule.setInitialWeight(1.0);
//        deltaBatchRule.setMaxIterate(10);
//        deltaBatchRule.buildClassifier(trainInstances);;
//        datasource = new DataSource(TESTDATASETJ48);
//        Instances testInstances = datasource.getDataSet();
//        testInstances.setClassIndex(testInstances.numAttributes()-1);
//        evaluation.evaluateModel(deltaBatchRule, testInstances);
//        System.out.print(evaluation.toMatrixString());
//        System.out.println(evaluation.toSummaryString());

        DeltaBatchRule deltaRulePerceptron = new DeltaBatchRule();
        deltaRulePerceptron.setMomentum(0);
        deltaRulePerceptron.setMaxIteration(10);
        deltaRulePerceptron.setLearningRate(0.1);
        deltaRulePerceptron.setTerminationDeltaMSE(1e-2);
        deltaRulePerceptron.buildClassifier(trainInstances);
        datasource = new DataSource(TRAINDATASET);
        Instances testInstances = datasource.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes()-1);
        evaluation.evaluateModel(deltaRulePerceptron, testInstances);
        System.out.print(evaluation.toMatrixString());
        System.out.println(evaluation.toSummaryString());

        // dari WEKA
//        J48 j48 = new J48();
//        j48.buildClassifier(trainInstances);
//        DataSource wekadatasource = new DataSource(TESTDATASETJ48);
//        Instances wekaTestInstances = wekadatasource.getDataSet();
//        wekaTestInstances.setClassIndex(wekaTestInstances.numAttributes()-1);
//        wekaEvaluation.evaluateModel(j48, wekaTestInstances);
//        System.out.print(evaluation.toMatrixString());
//        System.out.println(evaluation.toSummaryString());
    }
}
