package classifier.ann;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
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
    private double _terminationDeltaMSE;
    private int _nPredictor;

    private int[] _neuronPerHiddenLayer;
    private int[] _neuronPerLayer;
    private Boolean _isLinearOutput;
    private Neuron[][] _neuralNetwork;
    private List<Attribute> _predictorList;

    private NominalToBinary _nominalToBinary;

    public MultiLayerPerceptron() {
        // Initialization with default value
        _learningRate = 0.1;
        _momentum = 0.0;
        _terminationDeltaMSE = 1e-4;
        _maxIteration = 200;
        _neuronPerHiddenLayer = null;
        _neuronPerLayer = null;
        _isLinearOutput = null;
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
                    _neuralNetwork[layer][i].setWeights(generateRandomWeight(_nPredictor + 1));
                } else {
                    _neuralNetwork[layer][i].setWeights(generateRandomWeight(_neuronPerLayer[layer - 1] + 1));
                }
            }
        }

        // Change input and output to matrix
        _predictorList = new ArrayList<>();
        Enumeration attrIterator = numericInstances.enumerateAttributes();
        while (attrIterator.hasMoreElements()) {
            Attribute attr = (Attribute) attrIterator.nextElement();
            _predictorList.add(attr);
        }

        double[][] inputs = new double[numericInstances.numInstances()][_nPredictor + 1];
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
                outputs[instIndex][(int) instance.classValue()] = 1.0;
            } else if (numericInstances.classAttribute().isNumeric()) {
                outputs[instIndex][1] = instance.classValue();
            }
            inputs[instIndex][0] = 1.0;
            for (int i = 0; i < _predictorList.size(); i++) {
                inputs[instIndex][i + 1] = instance.value(_predictorList.get(i));
            }
        }

        // Learning

    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        return new double[0];
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

    public double getTerminationDeltaMSE() {
        return _terminationDeltaMSE;
    }

    public void setTerminationDeltaMSE(double terminationDeltaMSE) {
        _terminationDeltaMSE = terminationDeltaMSE;
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

    private double[] generateRandomWeight(int length) {
        double[] weights = new double[length];
        Random random = new Random();
        for (int i = 0; i < length; i++) {
            weights[i] = random.nextDouble();
        }
        return weights;
    }
}
