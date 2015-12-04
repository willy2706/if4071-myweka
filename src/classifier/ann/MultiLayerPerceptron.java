package classifier.ann;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.matrix.Maths;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

public class MultiLayerPerceptron extends Classifier {
    private double _learningRate;
    private int _maxIteration;
    private double _momentum;
    private double _terminationMseThreshold;
    private boolean _isVerbose;
    private int _nPredictor;
    private Double _initialWeight;
    private List<Attribute> _predictorList;

    private int[] _neuronPerHiddenLayer;
    private int[] _neuronPerLayer;
    private Boolean _isLinearOutput;
    private Neuron[][] _neuralNetwork;

    private NominalToBinary _nominalToBinary;

    private int _nIterationDone;

    public MultiLayerPerceptron() {
        // Initialization with default value
        _learningRate = 0.1;
        _momentum = 0.0;
        _terminationMseThreshold = 1e-4;
        _maxIteration = 200;
        _neuronPerHiddenLayer = null;
        _neuronPerLayer = null;
        _isLinearOutput = null;
        _nIterationDone = 0;
        _isVerbose = true;
        _initialWeight = null;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        // change all attr to numeric
        _nominalToBinary = new NominalToBinary();
        _nominalToBinary.setInputFormat(data);
        Instances numericInstances = Filter.useFilter(data, _nominalToBinary);

        _nPredictor = numericInstances.numAttributes() - 1;

        // Default neuron is one layer with number of predictor neurons
        if (_neuronPerHiddenLayer == null) {
            _neuronPerHiddenLayer = new int[]{_nPredictor};
        }
        _neuronPerLayer = new int[_neuronPerHiddenLayer.length + 1];
        System.arraycopy(_neuronPerHiddenLayer, 0, _neuronPerLayer, 0, _neuronPerHiddenLayer.length);
        if (numericInstances.classAttribute().isNominal()) {
            _neuronPerLayer[_neuronPerHiddenLayer.length] = numericInstances.classAttribute().numValues();
        } else {
            _neuronPerLayer[_neuronPerHiddenLayer.length] = 1;
        }

        // Build neural network with neurons
        if (_isLinearOutput == null) {
            _isLinearOutput = (numericInstances.classAttribute().isNumeric());
        }
        _neuralNetwork = new Neuron[_neuronPerLayer.length][];
        for (int layer = 0; layer < _neuronPerLayer.length; layer++) {
            _neuralNetwork[layer] = new Neuron[_neuronPerLayer[layer]];
            for (int i = 0; i < _neuronPerLayer[layer]; i++) {
                // Create with activation function
                if (layer == (_neuronPerLayer.length - 1) && _isLinearOutput) {
                    _neuralNetwork[layer][i] = new Neuron(Neuron.ActivationFunction.LINEAR);
                } else {
                    _neuralNetwork[layer][i] = new Neuron(Neuron.ActivationFunction.SIGMOID);
                }
                // Initialize weight
                if (layer == 0) {
                    if (_initialWeight == null) {
                        _neuralNetwork[layer][i].setWeights(generateRandomWeight(_nPredictor + 1));
                    } else {
                        _neuralNetwork[layer][i].setWeights(generateWeights(_initialWeight, _nPredictor + 1));
                    }
                } else {
                    if (_initialWeight == null) {
                        _neuralNetwork[layer][i].setWeights(generateRandomWeight(_neuronPerLayer[layer - 1] + 1));
                    } else {
                        _neuralNetwork[layer][i].setWeights(generateWeights(_initialWeight, _neuronPerLayer[layer - 1] + 1));
                    }
                }
            }
        }

        // Change input and output to matrix
        _predictorList = new ArrayList<Attribute>();
        Enumeration attrIterator = numericInstances.enumerateAttributes();
        while (attrIterator.hasMoreElements()) {
            Attribute attr = (Attribute) attrIterator.nextElement();
            _predictorList.add(attr);
        }

        double[][] inputs = new double[numericInstances.numInstances()][_nPredictor];
        double[][] outputs = null;
        if (numericInstances.classAttribute().isNominal()) {
            outputs = new double[numericInstances.numInstances()][numericInstances.classAttribute().numValues()];
            for (int i = 0; i < outputs.length; i++) {
                for (int j = 0; j < outputs[i].length; j++) {
                    outputs[i][j] = 0.0;
                }
            }
        } else if (numericInstances.classAttribute().isNumeric()) {
            outputs = new double[numericInstances.numInstances()][1];
        }

        for (int instIndex = 0; instIndex < numericInstances.numInstances(); instIndex++) {
            Instance instance = numericInstances.instance(instIndex);
            if (numericInstances.classAttribute().isNominal()) {
                int index = (int) instance.classValue();
                assert outputs != null;
                outputs[instIndex][index] = 1.0;
            } else if (numericInstances.classAttribute().isNumeric()) {
                assert outputs != null;
                outputs[instIndex][0] = instance.classValue();
            }
            for (int i = 0; i < _predictorList.size(); i++) {
                inputs[instIndex][i] = instance.value(_predictorList.get(i));
            }
        }

        // Learning Multi Layer Perceptron
        for (int iter = 0; iter < _maxIteration; iter++) {

            // Backprop learning
            for (int instIndex = 0; instIndex < inputs.length; instIndex++) {
                assert outputs != null;
                backpropUpdate(inputs[instIndex], outputs[instIndex]);
            }

            // Check termination condition
            _nIterationDone = iter + 1;
            double mseEvaluation = meanSquareErrorEvaluation(inputs, outputs);
            if (_isVerbose) {
                System.out.println("Epoch " + _nIterationDone + " MSE: " + mseEvaluation);
            }
            if (mseEvaluation < _terminationMseThreshold) break;

            // Output weights
            if (_isVerbose) {
                outputNeuronsWeights();
                System.out.println();
            }
        }

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MultiLayerPerceptron: cannot handle missing value");
        }
        _nominalToBinary.input(instance);
        Instance numericInstance = _nominalToBinary.output();
        double[] input = new double[_nPredictor];
        for (int i = 0; i < _predictorList.size(); i++) {
            input[i] = numericInstance.value(_predictorList.get(i));
        }

        // Predict
        double[] predicted = calculateOutput(input);

        // Remove minus value if nominal
        if (instance.classAttribute().isNominal()) {
            double pad = 0.0;
            for (int i = 0; i < predicted.length; i++) {
                if (predicted[i] < 0) {
                    pad = Math.max(pad, Math.abs(predicted[i]));
                }
            }
            for (int i = 0; i < predicted.length; i++) {
                predicted[i] += pad;
            }

            // Normalize, sum of predicted equals 1
            double sum = 0;
            for (int i = 0; i < predicted.length; i++) {
                sum += predicted[i];
            }
            if (sum > 0) {
                for (int i = 0; i < predicted.length; i++) {
                    predicted[i] /= sum;
                }
            }
        }

        return predicted;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    public double getLearningRate() {
        return _learningRate;
    }

    public void setLearningRate(double learningRate) {
        _learningRate = learningRate;
    }

    public int getMaxIteration() {
        return _maxIteration;
    }

    public void setMaxIteration(int maxIteration) {
        _maxIteration = maxIteration;
    }

    public double getMomentum() {
        return _momentum;
    }

    public void setMomentum(double momentum) {
        _momentum = momentum;
    }

    public double getTerminationMseThreshold() {
        return _terminationMseThreshold;
    }

    public void setTerminationMseThreshold(double terminationDeltaMSE) {
        _terminationMseThreshold = terminationDeltaMSE;
    }

    public boolean isLinearOutput() {
        return _isLinearOutput;
    }

    public void setIsLinearOutput(boolean isLinearOutput) {
        _isLinearOutput = isLinearOutput;
    }

    public int[] getNeuronPerHiddenLayer() {
        return _neuronPerHiddenLayer;
    }

    public void setNeuronPerHiddenLayer(int[] neuronPerHiddenLayer) {
        _neuronPerHiddenLayer = neuronPerHiddenLayer;
    }

    public int getEpochDone() {
        return _nIterationDone;
    }

    public boolean isVerbose() {
        return _isVerbose;
    }

    public void setIsVerbose(boolean isVerbose) {
        _isVerbose = isVerbose;
    }

    public double getInitialWeight() {
        return _initialWeight;
    }

    public void setInitialWeight(double initialWeight) {
        _initialWeight = initialWeight;
    }

    private void outputNeuronsWeights() {
        for (int layer = 0; layer < _neuralNetwork.length; layer++) {
            System.out.println("Layer " + layer);
            for (int neuronIdx = 0; neuronIdx < _neuralNetwork[layer].length; neuronIdx++) {
                System.out.print("    Neuron " + neuronIdx + " weights: ");
                double[] weights = _neuralNetwork[layer][neuronIdx].getWeights();
                for (int w = 0; w < weights.length; w++) {
                    System.out.print("" + w + ")" + weights[w] + " ");
                }
                System.out.println();
            }
        }
    }

    private double[] generateRandomWeight(int length) {
        double[] weights = new double[length];
        Random random = new Random();
        for (int i = 0; i < length; i++) {
            weights[i] = random.nextDouble();
        }
        return weights;
    }

    private double[] generateWeights(double value, int length) {
        double[] weights = new double[length];
        for (int i = 0; i < length; i++) {
            weights[i] = value;
        }
        return weights;
    }

    private double meanSquareErrorEvaluation(double[][] instancesInput, double[][] target) {
        double[][] predicted = new double[instancesInput.length][];
        // Calculate prediction
        for (int i = 0; i < instancesInput.length; i++) {
            predicted[i] = calculateOutput(instancesInput[i]);
        }

        // Calculate error
        double mse = 0.0;
        for (int inst = 0; inst < instancesInput.length; inst++) {
            double instSquareError = 0.0;
            for (int i = 0; i < target[i].length; i++) {
                instSquareError += Maths.square(target[inst][i] - predicted[inst][i]);
            }

            mse = mse + (instSquareError - mse) / (inst + 1);
        }

        return mse;
    }

    private double[] calculateOutput(double[] input) {
        double[] layerInput = input;
        double[] layerOutput = null;
        for (int layer = 0; layer < _neuronPerLayer.length; layer++) {
            layerOutput = new double[_neuronPerLayer[layer]];
            for (int i = 0; i < _neuronPerLayer[layer]; i++) {
                layerOutput[i] = _neuralNetwork[layer][i].calculateOutput(layerInput);
            }
            layerInput = layerOutput;
        }
        return layerOutput;
    }

    private void backpropUpdate(double[] input, double[] output) {
        recursiveBackprop(input, output, 0);
    }

    private double[] recursiveBackprop(double[] layerInput, double[] target, int layer) {
        // Calculate output
        double[] layerOutput = new double[_neuronPerLayer[layer]];
        for (int neuronIdx = 0; neuronIdx < _neuronPerLayer[layer]; neuronIdx++) {
            layerOutput[neuronIdx] = _neuralNetwork[layer][neuronIdx].calculateOutput(layerInput);
        }

        // Update error and recursive
        if (layer == (_neuronPerLayer.length - 1)) { // Output layer
            // Previous layer error
            double[] prevLayerNeuronError = new double[layerInput.length];
            for (int i = 0; i < prevLayerNeuronError.length; i++) prevLayerNeuronError[i] = 0.0;

            for (int neuronIdx = 0; neuronIdx < _neuronPerLayer[layer]; neuronIdx++) {
                double[] weights = _neuralNetwork[layer][neuronIdx].getWeights();
                double[] prevWeights = _neuralNetwork[layer][neuronIdx].getPrevWeights();
                if (prevWeights == null) {
                    prevWeights = new double[weights.length];
                    System.arraycopy(weights, 0, prevWeights, 0, weights.length);
                }
                double[] newWeights = new double[weights.length];

                double neuronError;
                if (_neuralNetwork[layer][neuronIdx].getActivationFunction() == Neuron.ActivationFunction.SIGMOID) {
                    neuronError = layerOutput[neuronIdx] * (1 - layerOutput[neuronIdx]) *
                            (target[neuronIdx] - layerOutput[neuronIdx]);
                } else {
                    neuronError = target[neuronIdx] - layerOutput[neuronIdx];
                }

                newWeights[0] = weights[0] + _learningRate * neuronError * 1 +
                        _momentum * (weights[0] - prevWeights[0]); // intercept
                for (int i = 0; i < layerInput.length; i++) {
                    newWeights[i + 1] = weights[i + 1] + _learningRate * neuronError * layerInput[i] +
                            +_momentum * (weights[i + 1] - prevWeights[i + 1]);
                    prevLayerNeuronError[i] += weights[i + 1] * neuronError;
                }
                _neuralNetwork[layer][neuronIdx].setWeights(newWeights);
            }

            // Return prev layer error
            return prevLayerNeuronError;

        } else {
            double[] eachNeuronError = recursiveBackprop(layerOutput, target, layer + 1);
            // Previous layer error
            double[] prevLayerNeuronError = new double[layerInput.length];
            for (int i = 0; i < prevLayerNeuronError.length; i++) prevLayerNeuronError[i] = 0.0;

            for (int neuronIdx = 0; neuronIdx < _neuronPerLayer[layer]; neuronIdx++) {
                double[] weights = _neuralNetwork[layer][neuronIdx].getWeights();
                double[] prevWeights = _neuralNetwork[layer][neuronIdx].getPrevWeights();
                if (prevWeights == null) {
                    prevWeights = new double[weights.length];
                    System.arraycopy(weights, 0, prevWeights, 0, weights.length);
                }
                double[] newWeights = new double[weights.length];

                double neuronError = layerOutput[neuronIdx] * (1 - layerOutput[neuronIdx]) * eachNeuronError[neuronIdx];

                newWeights[0] = weights[0] + _learningRate * neuronError * 1 +
                        _momentum * (weights[0] - prevWeights[0]); // intercept
                for (int i = 0; i < layerInput.length; i++) {
                    newWeights[i + 1] = weights[i + 1] + _learningRate * neuronError * layerInput[i] +
                            _momentum * (weights[i + 1] - prevWeights[i + 1]);
                    prevLayerNeuronError[i] += weights[i + 1] * neuronError;
                }
                _neuralNetwork[layer][neuronIdx].setWeights(newWeights);
            }

            // Return prev layer error
            return prevLayerNeuronError;
        }
    }

}
