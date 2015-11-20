package classifier.ann;


public class Neuron {

    private double[] _weights;
    private ActivationFunction _activationFunction;

    public Neuron(ActivationFunction activationFunction) {
        _activationFunction = activationFunction;
    }

    public double[] getWeights() {
        return _weights;
    }

    public void setWeights(double[] weights) {
        _weights = weights;
    }

    public ActivationFunction getActivationFunction() {
        return _activationFunction;
    }

    public double calculateOutput(double[] input) {
        // TODO implement
        return 0;
    }

    public void updateWeight(double error) {
        // TODO implement
    }

    public enum ActivationFunction {
        LINEAR,
        SIGMOID
    }

}
