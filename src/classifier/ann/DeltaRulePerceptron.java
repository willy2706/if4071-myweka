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
    protected int _nPredictor;
    protected double[] _initialWeights;
    protected double _learningRate;
    protected int _maxIteration;
    protected double _momentum;
    protected double _terminationDeltaMSE;
    protected List<Attribute> _predictorList;
    protected NominalToBinary _nominalToBinary;
    protected int _nIterationDone;
    protected double[] _prevWeight;
    protected double[] _lastWeight;
    private boolean _isVerbose;

    public DeltaRulePerceptron() {
        // Initialization with default value
        _learningRate = 0.1;
        _momentum = 0.0;
        _terminationDeltaMSE = 1e-4;
        _maxIteration = 200;
        _nIterationDone = 0;
        _isVerbose = false;
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

        // Initialize weight
        initWeight();

        // Change input to matrix
        _predictorList = new ArrayList<Attribute>();
        Enumeration attrIterator = numericInstances.enumerateAttributes();
        while (attrIterator.hasMoreElements()) {
            Attribute attr = (Attribute) attrIterator.nextElement();
            _predictorList.add(attr);
        }

        double[][] inputs = new double[numericInstances.numInstances()][_initialWeights.length];
        double[] targets = new double[numericInstances.numInstances()];

        for (int instIndex = 0; instIndex < numericInstances.numInstances(); instIndex++) {
            Instance instance = numericInstances.instance(instIndex);
            targets[instIndex] = instance.classValue();
            inputs[instIndex][0] = 1.0;
            for (int i = 0; i < _predictorList.size(); i++) {
                inputs[instIndex][i + 1] = instance.value(_predictorList.get(i));
            }
        }

        // Training Delta Rule Perceptron
        _prevWeight = null;
        _lastWeight = _initialWeights;
        double prevMse = meanSquareErrorEvaluation(inputs, targets);
        for (int it = 0; it < _maxIteration; it++) {

            for (int instIndex = 0; instIndex < inputs.length; instIndex++) {

                // Update weight
                double predicted = calculateOutput(inputs[instIndex]); //predicted adalah y
                double[] newWeight = new double[_initialWeights.length];
                for (int i = 0; i < newWeight.length; i++) {
                    double prevDeltaWeight;
                    if (it > 0 || instIndex > 0) {
                        prevDeltaWeight = _lastWeight[i] - _prevWeight[i];
                    } else {
                        prevDeltaWeight = 0;
                    }
                    double deltaWeight = _learningRate * (targets[instIndex] - predicted) * inputs[instIndex][i]
                            + (_momentum * prevDeltaWeight);
                    newWeight[i] = _lastWeight[i] + deltaWeight;
                }

                // Store update
                _prevWeight = _lastWeight;
                _lastWeight = newWeight;
            }

            _nIterationDone = it + 1;
            double mseEvaluation = meanSquareErrorEvaluation(inputs, targets);
            if (_isVerbose) {
                System.out.println("Epoch " + _nIterationDone + " MSE: " + mseEvaluation);
                System.out.println("Epoch " + _nIterationDone + " Delta MSE: " + (prevMse - mseEvaluation));
            }
            if (Math.abs(prevMse - mseEvaluation) < _terminationDeltaMSE) break;
            prevMse = mseEvaluation;

            // Output weight for each epoch
            if (_isVerbose) {
                System.out.print("Epoch " + _nIterationDone + " weights: ");
                for (int i = 0; i < _initialWeights.length; i++) {
                    System.out.print("" + i + ")" + _lastWeight[i] + " ");
                }
                System.out.println();
                System.out.println();
            }

        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("DeltaRulePerceptron: cannot handle missing value");
        }

        _nominalToBinary.input(instance);
        Instance numericInstance = _nominalToBinary.output();
        double[] input = new double[_initialWeights.length];
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
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    public void setInitialWeight(double[] weights) {
        _initialWeights = weights;
    }

    public double[] getInitialWeights() {
        return _initialWeights;
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

    public int getMaxIteration() {
        return _maxIteration;
    }

    public void setMaxIteration(int maxIteration) {
        this._maxIteration = maxIteration;
    }

    public double getLearningRate() {
        return _learningRate;
    }

    public void setLearningRate(double learningRate) {
        this._learningRate = learningRate;
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

    protected double calculateOutput(double[] input) {
        double output = 0.0;
        for (int i = 0; i < _nPredictor + 1; i++) {
            output += (_lastWeight[i] * input[i]);
        }
        return output;
    }

    protected double meanSquareErrorEvaluation(double[][] instancesInput, double[] target) {
        double[] predicted = new double[instancesInput.length];
        // Calculate prediction
        for (int i = 0; i < instancesInput.length; i++) {
            predicted[i] = calculateOutput(instancesInput[i]);
        }
        // Calculate error
        double mse = 0.0;
        for (int i = 0; i < instancesInput.length; i++) {
            mse = mse + (Maths.square (target[i] - predicted[i]) - mse) / (i + 1);
        }

        return mse;
    }

    protected void initWeight() {
        if (_initialWeights == null || _initialWeights.length != (_nPredictor + 1)) {
            _initialWeights = new double[_nPredictor + 1];
            Random random = new Random();
            for (int i = 0; i < _nPredictor + 1; i++) {
//                _initialWeights[i] = 0.0;
                _initialWeights[i] = random.nextDouble();
            }
        }
    }

}
