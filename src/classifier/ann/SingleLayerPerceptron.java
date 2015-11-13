package classifier.ann;

import common.util.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public class SingleLayerPerceptron{
    protected double learningRate;
    protected int maxIterate;
    protected double deltaMSE;
    protected double momentum;
    protected String initialWeight; //random or given

    public SingleLayerPerceptron() {
        
    };

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getMaxIterate() {
        return maxIterate;
    }

    public void setMaxIterate(int maxIterate) {
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
    
    public String getInitialWeight() {
        return initialWeight;
}

    public void setInitialWeight(String initialWeight) {
        this.initialWeight = initialWeight;
    }
}
