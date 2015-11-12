package classifier.ann;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public class PerceptronTrainingRule extends SingleLayerPerceptron{
    private double learningRate;
    private double maxIterate;
    private double deltaMSE;
    private double momentum;
    private String initialWeight; //random or given
    private String activationFunction;
    
    
    public PerceptronTrainingRule(int numInstance) {
        super(numInstance);
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
