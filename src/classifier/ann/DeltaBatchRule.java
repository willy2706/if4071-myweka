package classifier.ann;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.List;

/**
 * Created by nim_13512065 on 11/13/15.
 */
public class DeltaBatchRule extends MyANN {
    private List<InputValue> weight;
    private double learningRate;
    private int maxIterate;
    private double deltaMSE;
    private double momentum;
    private Double initialWeight; //random or given
    private Double[] weights;

    public DeltaBatchRule() {
        super();
    }

    private void initialWeight(int num) {
        setWeights(new Double[num]);
        for (int i = 0; i < num; ++i) {
            getWeights()[i] = getInitialWeight();
        }
    }

    private double calculateSumProduct(int cnt, double[] inputs) {
        double ret = 0.0;
        for (int i = 0; i < cnt; ++i) {
            ret+=inputs[i] * getWeights()[i];
        }
        return ret;
    }

    private double[] getInput(Instance instance) {
        double[] inputs = new double[instance.numAttributes()];
        for (int j = 0; j < instance.numAttributes(); ++j) {
            inputs[j] = instance.value(j);
        }
        return inputs;
    }

    @Override
    public void buildClassifier(Instances data) {
        initialWeight(data.numAttributes());
        int it = 1;
        double err = Double.MAX_VALUE;
        while (Utils.smOrEq(it,getMaxIterate()) && Utils.gr(err,getDeltaMSE())) {
            for (int i = 0; i < data.numInstances(); ++i) {
                Instance row = data.instance(i);
                double[] inputs = getInput(row);
                double y = calculateSumProduct(data.numAttributes(), inputs);
                //TODO hitung MSE
            }
//            jangan lupa set deltamse
//            setDeltaMSE();
            ++it;
        }
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

    public Double[] getWeights() {
        return weights;
    }

    public void setWeights(Double[] weights) {
        this.weights = weights;
    }

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

    public Double getInitialWeight() {
        return initialWeight;
    }

    public void setInitialWeight(Double initialWeight) {
        this.initialWeight = initialWeight;
    }
}
