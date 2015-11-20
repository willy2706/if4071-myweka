package classifier.ann;

import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

/**
 * Created by nim_13512065 on 11/13/15.
 */
public class DeltaBatchRule extends DeltaRulePerceptron {

    public DeltaBatchRule() {
        super();
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

            double[] deltaWeight = new double[_initialWeights.length];
            double[] newWeight = new double[_initialWeights.length];
            for (int instIndex = 0; instIndex < inputs.length; instIndex++) {

                // calculate delta weight
                double predicted = calculateOutput(inputs[instIndex]); //predicted adalah y

                for (int i = 0; i < deltaWeight.length; i++) {
                    double prevDeltaWeight;
                    if (it > 0) {
                        prevDeltaWeight = _lastWeight[i] - _prevWeight[i];
                    } else {
                        prevDeltaWeight = 0;
                    }
                    deltaWeight[i] +=  _learningRate * (targets[instIndex] - predicted) * inputs[instIndex][i]
                            + (_momentum * prevDeltaWeight);
                    if (instIndex == inputs.length-1) {
                        newWeight[i] = _lastWeight[i] + deltaWeight[i];
                    }
                }
            }
            // Store update
            _prevWeight = _lastWeight;
            _lastWeight = newWeight;

            _nIterationDone = it + 1;
            double mseEvaluation = meanSquareErrorEvaluation(inputs, targets);
            System.out.println("Epoch " + _nIterationDone + " MSE: " + mseEvaluation);
            System.out.println("Epoch " + _nIterationDone + " Delta MSE: " + (prevMse - mseEvaluation));
            if (Math.abs(prevMse - mseEvaluation) < _terminationDeltaMSE) break;
            prevMse = mseEvaluation;

            // Output weight for each epoch
            System.out.print("Epoch " + _nIterationDone + " Weight: ");
            for (int i = 0; i < _initialWeights.length; i++) {
                System.out.print("" + i + ")" + _lastWeight[i] + " ");
            }
            System.out.println();

        }
    }
}
