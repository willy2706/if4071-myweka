/**
 * Created by nim_13512065 on 9/24/15.
 */


import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public final static String TRAINDATASETJ48 = "<isisendiri>.arff";
    public final static String TESTDATASETJ48 = "<isisendiri>.arff";
    public static void main(String[] args) throws Exception {
        DataSource datasource = new DataSource(TRAINDATASETJ48);
        Instances trainInstances = datasource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        Evaluation evaluation = new Evaluation(trainInstances);

        MyJ48 myJ48 = new MyJ48();

        datasource = new DataSource(TESTDATASETJ48);
        Instances testInstances = null;

        evaluation.evaluateModel(myJ48,testInstances);
    }
}
