package classifier.ann;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public class PerceptronTrainingRule extends MyANN {
    public PerceptronTrainingRule() {
        setHiddenLayers("0");
    }

    @Override
    public void buildClassifier(Instances data) {
        //TODO
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        //TODO
        return new double[0];
    }

    @Override
    public double classifyInstance(Instance instance) {
        //TODO
        return 0;
    }

    @Override
    public Capabilities getCapabilities() {
        //TODO
        return null;
    }
}
