package classifier.ann;

import weka.core.Instance;
import weka.core.Instances;

public class MulticlassDeltaRulePerceptron extends SinglePerceptron {
    private Double _initialWeight = null;
    private MultiLayerPerceptron multiLayerPerceptron = null;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        multiLayerPerceptron = new MultiLayerPerceptron();
        multiLayerPerceptron.buildClassifier(data);
        multiLayerPerceptron.setIsLinearOutput(true);
        multiLayerPerceptron.setNeuronPerHiddenLayer(null);
        multiLayerPerceptron.setLearningRate(getLearningRate());
        multiLayerPerceptron.setMomentum(getMomentum());
        if (getInitialWeight() != null) multiLayerPerceptron.setInitialWeight(getInitialWeight());
        multiLayerPerceptron.buildClassifier(data);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return multiLayerPerceptron.distributionForInstance(instance);
    }

    public Double getInitialWeight() {
        return _initialWeight;
    }

    public void setInitialWeight(Double weight) {
        _initialWeight = weight;
    }
}
