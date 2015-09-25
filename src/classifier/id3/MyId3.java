package classifier.id3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class MyId3 extends Classifier {

    private static final long serialVersionUID = 3658199532232755963L;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // TODO implement
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // TODO implement
        return 0.0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        // TODO implement
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    public String toString() {
        return "MyId3 is implemented based on Id3 for educational purpose";
    }
}
