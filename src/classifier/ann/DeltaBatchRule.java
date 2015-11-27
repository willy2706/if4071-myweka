package classifier.ann;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

import java.util.ArrayList;
import java.util.Enumeration;

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
        _lastWeight = _initialWeights;
        double[] prevDeltaWeight = null;
        double[] deltaWeight = null;
        for (int it = 0; it < _maxIteration; it++) {
            if (it > 0) {
                prevDeltaWeight = deltaWeight;
            } else {
                prevDeltaWeight = new double[_initialWeights.length];
                for (int z = 0; z < prevDeltaWeight.length; ++z) {
                    prevDeltaWeight[z] = 0.0;
                }
            }
            double[] newWeight = new double[_initialWeights.length];

            deltaWeight = new double[_initialWeights.length];
            for (int z = 0; z < deltaWeight.length; ++z) {
                deltaWeight[z] = 0.0;
            }

            for (int instIndex = 0; instIndex < inputs.length; instIndex++) {
                // calculate delta weight
                double predicted = calculateOutput(inputs[instIndex]); //predicted adalah y

                for (int i = 0; i < deltaWeight.length; i++) {
                    deltaWeight[i] +=  _learningRate * (targets[instIndex] - predicted) * inputs[instIndex][i]
                            + (_momentum * prevDeltaWeight[i]);
                    if (instIndex == inputs.length-1) {
                        newWeight[i] = _lastWeight[i] + deltaWeight[i];
                    }
                }
            }
            // Store update
            _lastWeight = newWeight;

            _nIterationDone = it + 1;
            double mseEvaluation = meanSquareErrorEvaluation(inputs, targets);
            System.out.println("Epoch " + _nIterationDone + " MSE: " + mseEvaluation);
            if (mseEvaluation < _terminationMseThreshold) break;

            // Output weight for each epoch
            System.out.print("Epoch " + _nIterationDone + " Weight: ");
            for (int i = 0; i < _initialWeights.length; i++) {
                System.out.print("" + i + ")" + _lastWeight[i] + " ");
            }
            System.out.println();

        }
    }
}
