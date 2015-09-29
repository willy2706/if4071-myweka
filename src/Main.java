/**
 * Created by nim_13512065 on 9/24/15.
 */


import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
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


//        /*sample*/
        J48 classifier = new J48();
//        classifier.buildClassifier(trainInstances);
//        evaluation.evaluateModel(classifier, trainInstances);


        /*what we will use*/
        MyJ48 myJ48 = new MyJ48();
        myJ48.buildClassifier(trainInstances);
        datasource = new DataSource(TESTDATASETJ48);
        Instances testInstances = datasource.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes()-1);
        evaluation.evaluateModel(myJ48, testInstances);
        System.out.print(evaluation.toMatrixString());
        System.out.println(evaluation.toSummaryString());
        
        // dari WEKA
        J48 j48 = new J48();
        j48.buildClassifier(trainInstances);
        DataSource wekadatasource = new DataSource(TESTDATASETJ48);
        Instances wekaTestInstances = wekadatasource.getDataSet();
        wekaTestInstances.setClassIndex(wekaTestInstances.numAttributes()-1);
        wekaEvaluation.evaluateModel(j48, wekaTestInstances);
        System.out.print(evaluation.toMatrixString());
        System.out.println(evaluation.toSummaryString());
    }
}
