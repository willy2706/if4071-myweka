package classifier.ann;

import common.util.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public class SingleLayerPerceptron{
    protected List<InputValue> weight;
    protected double learningRate;
    protected double maxIterate;
    protected double deltaMSE;
    protected double momentum;
    protected String initialWeight; //random or given
    protected ActivationFunction activationFunction;

    public SingleLayerPerceptron() {
        weight = new ArrayList<InputValue>();
    };

    public List<InputValue> getWeight() {
        return weight;
    }

    public void setWeight(List<InputValue> weight) {
        this.weight = weight;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getMaxIterate() {
        return maxIterate;
    }

    public void setMaxIterate(double maxIterate) {
        this.maxIterate = maxIterate;
    }

    public double getDeltaMSE() {
        return deltaMSE;
    }

    public void setDeltaMSE(double deltaMSE) {
        this.deltaMSE = deltaMSE;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public String getInitialWeight() {
        return initialWeight;
    }

    public void setInitialWeight(String initialWeight) {
        this.initialWeight = initialWeight;
    }
}
