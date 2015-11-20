/**
 * Created by nim_13512065 on 9/24/15.
 */


import classifier.ann.DeltaRulePerceptron;
import classifier.ann.PerceptronTrainingRule;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public final static String FOLDER = "dataset/";
    public final static String TRAINDATASETJ48 = FOLDER + "weather.nominal.arff";
    public final static String TESTDATASETJ48 = FOLDER + "test.weather.nominal.arff";
    public static void main(String[] args) throws Exception {
        DataSource datasource = new DataSource(TRAINDATASETJ48);
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

//        DeltaRulePerceptron deltaRulePerceptron = new DeltaRulePerceptron();
//        deltaRulePerceptron.setMomentum(0.001);
//        deltaRulePerceptron.setMaxIteration(100);
//        deltaRulePerceptron.setLearningRate(0.01);
//        deltaRulePerceptron.setTerminationDeltaMSE(1e-10);
//        deltaRulePerceptron.buildClassifier(trainInstances);
//        datasource = new DataSource(TRAINDATASETJ48);
//        Instances testInstances = datasource.getDataSet();
//        testInstances.setClassIndex(testInstances.numAttributes()-1);
//        evaluation.evaluateModel(deltaRulePerceptron, testInstances);
//        System.out.print(evaluation.toMatrixString());
//        System.out.println(evaluation.toSummaryString());
        
        PerceptronTrainingRule perceptronTrainingRule = new PerceptronTrainingRule();
        perceptronTrainingRule.setMomentum(0.001);
        perceptronTrainingRule.setMaxIteration(100);
        perceptronTrainingRule.setLearningRate(0.0001);
        perceptronTrainingRule.setTerminationDeltaMSE(1e-10);
        perceptronTrainingRule.buildClassifier(trainInstances);
        datasource = new DataSource(TRAINDATASETJ48);
        Instances testInstances = datasource.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes()-1);
        evaluation.evaluateModel(perceptronTrainingRule, testInstances);
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
