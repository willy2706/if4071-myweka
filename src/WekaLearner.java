import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Created by Winson on 9/26/2015.
 */
public class WekaLearner {

    public static final String[] CLASSIFIER_AVAILABLE = {"J48", "NaiveBayes", "IBk", "MultilayerPerceptron"};
    private Classifier _classifier;
    private Instances _instances;
    private int _classIndex;
    private Evaluation _evaluation;

    public WekaLearner() throws Exception {

    }

    public void setClassifier(Classifier classifier) throws Exception {
        _classifier = classifier;
    }

    public final Instances getTrainningData() {
        return _instances;
    }

    public void setTrainningData(String fileLocation) throws Exception {
        // Default class index is last column
        _instances = ConverterUtils.DataSource.read(fileLocation);
        _classIndex = _instances.numAttributes() - 1;
        _instances.setClassIndex(_classIndex);
        _evaluation = new Evaluation(_instances);
    }

    public void setClassIndex(int index) {
        _classIndex = index;
        _instances.setClassIndex(_classIndex);
    }

    public void removeAttribute(int index[]) throws Exception {
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(index);
        _instances = Filter.useFilter(_instances, removeFilter);
    }

    public void resampleInstances(double sampleSizePercentage, boolean withReplacement) throws Exception {
        Resample resampleFilter = new Resample();
        resampleFilter.setNoReplacement(!withReplacement);
        resampleFilter.setSampleSizePercent(sampleSizePercentage);
        _instances = Filter.useFilter(_instances, resampleFilter);
    }

    public double classify(double[] instance) throws Exception {
        _classifier.buildClassifier(_instances);
        Instances ins = new Instances(_instances, 0);
        Instance row = new Instance(1.0, instance);
        ins.add(row);
        return _classifier.classifyInstance(ins.lastInstance());
    }

    public String fullTranningEvaluation() throws Exception {
        _classifier.buildClassifier(_instances);

        _evaluation = new Evaluation(_instances);
        _evaluation.evaluateModel(_classifier, _instances);

        return _classifier.toString() + _evaluation.toSummaryString("\nHasil evaluasi dengan full-trainning:\n", false);
    }

    public String crossValidationEvaluation(int fold) throws Exception {
        _evaluation = new Evaluation(_instances);
        _evaluation.crossValidateModel(_classifier, _instances, fold, _instances.getRandomNumberGenerator(1));

        return _classifier.toString() + _evaluation.toSummaryString("\nHasil evaluasi dengan cross-validation " + Integer.toString(fold) + "-fold:\n", false);
    }

    public void loadModel(String fileLocation) throws Exception {
        _classifier = (Classifier) SerializationHelper.read(fileLocation);
    }

    public void saveModel(String fileLocation) throws Exception {
        SerializationHelper.write(fileLocation, _classifier);
    }


}
