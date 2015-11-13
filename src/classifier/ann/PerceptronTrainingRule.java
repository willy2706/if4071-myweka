package classifier.ann;

import common.util.ActivationFunction;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public class PerceptronTrainingRule extends MyANN{
    private SingleLayerPerceptron perceptron;
    private int numAttributes;
    private int numInstances;

    public int getNumAttributes() {
        return numAttributes;
    }

    public void setNumAttributes(int numAttributes) {
        this.numAttributes = numAttributes;
    }

    public int getNumInstances() {
        return numInstances;
    }

    public void setNumInstances(int numInstances) {
        this.numInstances = numInstances;
    }
    
    public PerceptronTrainingRule(){
        perceptron = new SingleLayerPerceptron();
    }

    @Override
    public void buildClassifier(Instances data) {
        numInstances = data.numInstances();
        numAttributes = data.numAttributes();
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
