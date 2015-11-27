package classifier.ann;

import weka.classifiers.Classifier;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public abstract class SinglePerceptron extends Classifier {
    protected boolean verbose;
    protected double learningRate;
    protected int maxIteration;
    protected double momentum;
    protected double terminationMseThreshold;

    public boolean isVerbose() {
        return verbose;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getMaxIteration() {
        return maxIteration;
    }

    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public double getTerminationMseThreshold() {
        return terminationMseThreshold;
    }

    public void setTerminationMseThreshold(double terminationMseThreshold) {
        this.terminationMseThreshold = terminationMseThreshold;
    }

}
