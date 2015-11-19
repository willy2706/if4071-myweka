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

public class DeltaRulePerceptron extends Classifier {
    private int _nPredictor;
    private double[] _weights;
    private double _learningRate;
    private int _maxIteration;
    private double _momentum;
    private double _terminationDeltaMSE;
    private List<Attribute> _predictorList;

    private NominalToBinary _nominalToBinary;

    private int _nIterationDone;
    private List<double[]> _updatedWeights;

    public DeltaRulePerceptron() {

        // Initialization with default value
        _learningRate = 0.1;
        _momentum = 0.0;
        _terminationDeltaMSE = 0.001;
        _maxIteration = 200;
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
        Instances numericInstances = Filter.useFilter(data, _nominalToBinary);

        _nPredictor = numericInstances.numAttributes() - 1;

        // Initialize weight
        initWeight();

        // Change input to matrix
        _predictorList = new ArrayList<>();
        Enumeration attrIterator = numericInstances.enumerateAttributes();
        while (attrIterator.hasMoreElements()) {
            Attribute attr = (Attribute) attrIterator.nextElement();
            _predictorList.add(attr);
        }

        double[][] inputs = new double[numericInstances.numInstances()][_weights.length];
        double[] outputs = new double[numericInstances.numInstances()];

        for (int instIndex = 0; instIndex < numericInstances.numInstances(); instIndex++) {
            Instance instance = numericInstances.instance(instIndex);
            outputs[instIndex] = instance.classValue();
            inputs[instIndex][0] = 1.0;
            for (int i = 0; i < _predictorList.size(); i++) {
                inputs[instIndex][i + 1] = instance.value(_predictorList.get(i));
            }
        }

        // Training Delta Rule Perceptron
        _nIterationDone = 0;
        _updatedWeights = new ArrayList<>();
        // updated weights index 0 is initial weight
        _updatedWeights.add(_weights);
        double prevMse = meanSquareErrorEvaluation(inputs, outputs);
        for (int it = 0; it < _maxIteration; it++) {

            for (int instIndex = 0; instIndex < inputs.length; instIndex++) {

                // Update weight
                double predicted = calculateOutput(inputs[instIndex]);
                double[] newWeight = new double[_weights.length];
                for (int i = 0; i < newWeight.length; i++) {
                    double prevDeltaWeight;
                    if (_nIterationDone > 0) {
                        prevDeltaWeight = _updatedWeights.get(_nIterationDone)[i] - _updatedWeights.get(_nIterationDone - 1)[i];
                    } else {
                        prevDeltaWeight = 0;
                    }
                    double deltaWeight = _learningRate * (outputs[instIndex] - predicted) * inputs[instIndex][i]
                            + _momentum * prevDeltaWeight;
                    newWeight[i] += deltaWeight;
                }

                // Store update
                _nIterationDone++;
                _updatedWeights.add(newWeight);
            }

            double mseEvaluation = meanSquareErrorEvaluation(inputs, outputs);
            // TODO use absolut or not?
            if ((prevMse - mseEvaluation) < _terminationDeltaMSE) break;
            prevMse = mseEvaluation;

            // Output weight for each epoch
            // TODO use system.out or logging
            for (int i = 0; i < _weights.length; i++) {
                System.out.print("" + i + ":" + _updatedWeights.get(_nIterationDone)[i] + " ");
            }
            System.out.printf("\n");

        }

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("DeltaRulePerceptron: cannot handle missing value");
        }

        _nominalToBinary.input(instance);
        Instance numericInstance = _nominalToBinary.output();
        double[] input = new double[_weights.length];
        input[0] = 1.0;
        for (int i = 0; i < _predictorList.size(); i++) {
            input[i + 1] = numericInstance.value(_predictorList.get(i));
        }
        double prediction = calculateOutput(input);
        if (numericInstance.classAttribute().isNumeric()) {
            return new double[]{prediction};
        } else if (numericInstance.classAttribute().isNominal()) {
            if (prediction < 0.0) prediction = 0.0;
            if (prediction > 1.0) prediction = 1.0;
            double class0 = 1.0 - prediction;
            double class1 = prediction;
            return new double[]{class0, class1};
        } else {
            return null;
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    public void setInitialWeight(double[] weights) {
        _weights = weights;
    }

    public double[] getWeights() {
        return _weights;
    }

    public double getTerminationDeltaMSE() {
        return _terminationDeltaMSE;
    }

    public void setTerminationDeltaMSE(double terminationDeltaMSE) {
        this._terminationDeltaMSE = terminationDeltaMSE;
    }

    public double getMomentum() {
        return _momentum;
    }

    public void setMomentum(double momentum) {
        this._momentum = momentum;
    }

    public void setMaxIteration(int maxIteration) {
        this._maxIteration = maxIteration;
    }

    public void setLearningRate(double learningRate) {
        this._learningRate = learningRate;
    }

    private double calculateOutput(double[] input) {
        double output = 0.0;
        for (int i = 0; i < _nPredictor + 1; i++) {
            output += _updatedWeights.get(_nIterationDone)[i] * input[i];
        }
        return output;
    }

    private double meanSquareErrorEvaluation(double[][] instancesInput, double[] target) {
        double[] predicted = new double[instancesInput.length];
        // Calculate prediction
        for (int i = 0; i < instancesInput.length; i++) {
            predicted[i] = calculateOutput(instancesInput[i]);
        }
        // Calculate error
        double mse = 0.0;
        for (int i = 0; i < instancesInput.length; i++) {
            mse += Maths.square(target[i] - predicted[i]);
        }
        mse /= instancesInput.length;

        return mse;
    }

    private void initWeight() {
        if (_weights.length != (_nPredictor + 1)) {
            _weights = new double[_nPredictor + 1];
            Random random = new Random();
            for (int i = 0; i < _nPredictor + 1; i++) {
                _weights[i] = (random.nextDouble() % 1000.0);
            }
        }
    }

}
