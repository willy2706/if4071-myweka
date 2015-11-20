package classifier.ann;

import common.util.ActivationFunction;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public class PerceptronTrainingRule extends MyANN{
    private SingleLayerPerceptron perceptron;
    private int numAttributes;
    private double[] weights;
    private List<double[]> inputs; //matrix input
    private ActivationFunction af= ActivationFunction.SIGN;

    public int getNumAttributes() {
        return numAttributes;
    }

    public void setNumAttributes(int numAttributes) {
        this.numAttributes = numAttributes;
    }

    public PerceptronTrainingRule(){
        perceptron = new SingleLayerPerceptron();
    }

    @Override
    public void buildClassifier(Instances data) {
        numAttributes = data.numAttributes();
        
        //initialize weight
        weights = new double[numAttributes+1];
        for(int i=0;i<numAttributes;++i) {
            weights[i] = 0;
        }
        
        //change input to matrix
        inputs = new ArrayList<double[]>();
        double[] targets = new double[data.numInstances()+1];
        Enumeration instanceIterator = data.enumerateInstances();
        int it=0;
        int classIndex = data.classIndex();
        while(instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            targets[it] = instance.classValue();
            double[] input = new double[numAttributes];
            for(int i=0;i<numAttributes;++i) {
                input[i] = instance.value(i);
            }
            inputs.add(input);
            it++;
        }
        
        //training
        for(int i=0;i<perceptron.getMaxIterate();++i) {
            for(int j=0;j<inputs.size();++j) {
                double output = calculateSum(inputs.get(j));
                output = af.calculateOutput(output);
                for(int k=0;k<inputs.get(j).length;k++) {
                    updateWeight(targets[j],output,inputs.get(j)[k]);
                }
            }
        }
    }
    
    public double calculateSum(double[] inputs) {
        double out = 0.0;
        for(int i=0;i<inputs.length+1;++i) {
            out+= inputs[i] * weights[i];
        }
        return out;
    }
    
    private void updateWeight(double target, double output, double input) {
        double[] newWeight = new double[numAttributes+1];
        for(int i=0;i<inputs.size();i++) {
            newWeight[i] = weights[i]+ perceptron.getLearningRate()*(target-output)*input;
        }
        weights = newWeight;
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        return new double[0];
    }

    @Override
    public double classifyInstance(Instance instance) {
        return 0;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
