/**
 * Created by nim_13512065 on 9/24/15.
 */

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public final static String FOLDER = "myweka/dataset/";
    public final static String TRAINDATASETJ48 = FOLDER + "weather.nominal.arff";
    public final static String TESTDATASETJ48 = FOLDER + "<isisendiri>.arff";
    public static void main(String[] args) throws Exception {
        DataSource datasource = new DataSource(TRAINDATASETJ48);
        Instances trainInstances = datasource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        Evaluation evaluation = new Evaluation(trainInstances);

//        /*sample*/
//        J48 classifier = new J48();
//        classifier.buildClassifier(trainInstances);
//        evaluation.evaluateModel(classifier, trainInstances);


        /*what we will use*/
        MyJ48 myJ48 = new MyJ48();
        myJ48.buildClassifier(trainInstances);
//        datasource = new DataSource(TESTDATASETJ48);
//        Instances testInstances = datasource.getDataSet();
        evaluation.evaluateModel(myJ48,trainInstances);
        System.out.print(evaluation.toSummaryString());
    }
}
