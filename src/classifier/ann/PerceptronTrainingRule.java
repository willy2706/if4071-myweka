package classifier.ann;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public class PerceptronTrainingRule extends SingleLayerPerceptron{
    
    public PerceptronTrainingRule(int numInstance) {
        super(numInstance);
        setHiddenLayers("0");
    }

    @Override
    public void buildClassifier(Instances data) {
        
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        return new double[0];
    }

    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
