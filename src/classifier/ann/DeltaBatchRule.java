package classifier.ann;

import weka.core.*;
import weka.core.matrix.Maths;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

/**
 * Created by nim_13512065 on 11/13/15.
 */
public class DeltaBatchRule extends SinglePerceptron {

    private int nPredictor;
    private double[] initialWeight;
    private List<Attribute> predictorList;
    private NominalToBinary nominalToBinary;
    private int nIterationDone;
    private double[] lastWeight;

    public DeltaBatchRule() {
        super();
        setnIterationDone(0);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        // change all attr to numeric
        setNominalToBinary(new NominalToBinary());
        getNominalToBinary().setInputFormat(data);
        Instances numericInstances = Filter.useFilter(data, getNominalToBinary());

        setnPredictor(numericInstances.numAttributes() - 1);

        // Initialize weight
        initWeight();

        // Change input to matrix
        setPredictorList(new ArrayList<Attribute>());
        Enumeration attrIterator = numericInstances.enumerateAttributes();
        while (attrIterator.hasMoreElements()) {
            Attribute attr = (Attribute) attrIterator.nextElement();
            getPredictorList().add(attr);
        }

        double[][] inputs = new double[numericInstances.numInstances()][getInitialWeight().length];
        double[] targets = new double[numericInstances.numInstances()];

        for (int instIndex = 0; instIndex < numericInstances.numInstances(); instIndex++) {
            Instance instance = numericInstances.instance(instIndex);
            targets[instIndex] = instance.classValue();
            inputs[instIndex][0] = 1.0;
            for (int i = 0; i < getPredictorList().size(); i++) {
                inputs[instIndex][i + 1] = instance.value(getPredictorList().get(i));
            }
        }

        // Training Delta Rule Perceptron
        setLastWeight(getInitialWeight());
        double[] prevDeltaWeight = null;
        double[] deltaWeight = null;
        for (int it = 0; it < getMaxIteration(); it++) {
            if (it > 0) {
                prevDeltaWeight = deltaWeight;
            } else {
                prevDeltaWeight = new double[getInitialWeight().length];
                for (int z = 0; z < prevDeltaWeight.length; ++z) {
                    prevDeltaWeight[z] = 0.0;
                }
            }
            double[] newWeight = new double[getInitialWeight().length];

            deltaWeight = new double[getInitialWeight().length];
            for (int z = 0; z < deltaWeight.length; ++z) {
                deltaWeight[z] = 0.0;
            }

            for (int instIndex = 0; instIndex < inputs.length; instIndex++) {
                // calculate delta weight
                double predicted = calculateOutput(inputs[instIndex]); //predicted adalah y

                for (int i = 0; i < deltaWeight.length; i++) {
                    deltaWeight[i] +=  getLearningRate() * (targets[instIndex] - predicted) * inputs[instIndex][i]
                            + (getMomentum() * prevDeltaWeight[i]);
                    if (instIndex == inputs.length-1) {
                        newWeight[i] = getLastWeight()[i] + deltaWeight[i];
                    }
                }
            }
            // Store update
            setLastWeight(newWeight);

            setnIterationDone(it + 1);
            double mseEvaluation = meanSquareErrorEvaluation(inputs, targets);
            System.out.println("Epoch " + getnIterationDone() + " MSE: " + mseEvaluation);
            if (mseEvaluation < getTerminationMseThreshold()) break;

            // Output weight for each epoch
            System.out.print("Epoch " + getnIterationDone() + " Weight: ");
            for (int i = 0; i < getInitialWeight().length; i++) {
                System.out.print("" + i + ")" + getLastWeight()[i] + " ");
            }
            System.out.println();

        }
    }
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("DeltaRulePerceptron: cannot handle missing value");
        }

        getNominalToBinary().input(instance);
        Instance numericInstance = getNominalToBinary().output();
        double[] input = new double[getInitialWeight().length];
        input[0] = 1.0;
        for (int i = 0; i < getPredictorList().size(); i++) {
            input[i + 1] = numericInstance.value(getPredictorList().get(i));
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

    private double calculateOutput(double[] input) {
        double output = 0.0;
        for (int i = 0; i < getnPredictor() + 1; i++) {
            output += (getLastWeight()[i] * input[i]);
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
            mse = mse + (Maths.square (target[i] - predicted[i]) - mse) / (i + 1);
        }

        return mse;
    }

    private void initWeight() {
        if (getInitialWeight() == null || getInitialWeight().length != (getnPredictor() + 1)) {
            setInitialWeight(new double[getnPredictor() + 1]);
            Random random = new Random();
            for (int i = 0; i < getnPredictor() + 1; i++) {
//               initialWeight[i] = 0.0;
                getInitialWeight()[i] = random.nextDouble();
            }
        }
    }

    private int getnPredictor() {
        return nPredictor;
    }

    private void setnPredictor(int nPredictor) {
        this.nPredictor = nPredictor;
    }

    public double[] getInitialWeight() {
        return initialWeight;
    }

    public void setInitialWeight(double[] initialWeight) {
        this.initialWeight = initialWeight;
    }

    private List<Attribute> getPredictorList() {
        return predictorList;
    }

    private void setPredictorList(List<Attribute> predictorList) {
        this.predictorList = predictorList;
    }

    private NominalToBinary getNominalToBinary() {
        return nominalToBinary;
    }

    private void setNominalToBinary(NominalToBinary nominalToBinary) {
        this.nominalToBinary = nominalToBinary;
    }

    private int getnIterationDone() {
        return nIterationDone;
    }

    private void setnIterationDone(int nIterationDone) {
        this.nIterationDone = nIterationDone;
    }

    private double[] getLastWeight() {
        return lastWeight;
    }

    private void setLastWeight(double[] lastWeight) {
        this.lastWeight = lastWeight;
    }
}
