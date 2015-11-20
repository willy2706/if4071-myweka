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
        double output = _weights[0];
        for (int predIdx = 0; predIdx < input.length; predIdx++) {
            output += _weights[predIdx + 1] * input[predIdx];
        }

        if (_activationFunction == ActivationFunction.SIGMOID) {
            return 1.0 / (1.0 + Math.exp(-output));
        } else { // Linear Output, same with no using activation function
            return output;
        }
    }

    public enum ActivationFunction {
        LINEAR,
        SIGMOID
    }

}
