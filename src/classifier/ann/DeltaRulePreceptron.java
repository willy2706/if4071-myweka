package classifier.ann;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Maths;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

public class DeltaRulePreceptron extends Classifier{
    private int _nAttribute;
    private double[] _weights;
    private double learningRate; // TODO from where input and initialize
    private double maxIteration; // TODO from where input and initialize

    @Override
    public void buildClassifier(Instances data) throws Exception {
        _nAttribute = data.numAttributes();

        // Initialize weight
        _weights = new double[_nAttribute+1];
        Random random = new Random();
        for(int i=0; i<_nAttribute; i++){
            _weights[i] = (random.nextDouble() % 1000.0);
        }

        // Change input to matrix
        List<double[]> inputs = new ArrayList<>();
        double[] output = new double[data.numInstances()];
        Enumeration instanceIterator = data.enumerateInstances();
        int count = 0;
        int classIndex = data.classIndex();
        while (instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            output[count] = instance.classValue();
            double input[] = new double[_nAttribute];
            int c = 0;
            for(int i=0; i<instance.numAttributes(); i++){
                if (i != classIndex){
                    input[c] = instance.value(i);
                    c++;
                }
            }
            inputs.add(input);
            count++;
        }

        // Training
        for(int it=0; it<maxIteration; it++){
            for(int i=0; i<inputs.size(); i++){
                double out = calculateOutput(inputs.get(i));

            }
        }


    }

    private double calculateOutput(double[] input){
        double output = 0.0;
        for(int i=0; i<_nAttribute+1; i++){
            output += _weights[i] * input[i];
        }
        return output;
    }

    private void updateWeight(double target, double output, double input){
        double[] newWeight = new double[_nAttribute+1];
        for(int i=0; i<_nAttribute+1; i++){
            newWeight[i] = _weights[i] + learningRate * (target - output) * input;
        }
        _weights = newWeight;
    }

    private double meanSquareErrorEvaluation(List<double[]> instancesInput, double[] target){
        double[] predicted = new double[instancesInput.size()];
        // Calculate prediction
        for(int i=0; i<instancesInput.size(); i++){
            predicted[i] = calculateOutput(instancesInput.get(i));
        }
        // Calculate error
        double mse =0.0;
        for(int i=0; i<instancesInput.size(); i++){
            mse += Maths.square(target[i] - predicted[i]);
        }
        mse /= instancesInput.size();

        return mse;
    }
}
