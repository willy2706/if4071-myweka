package classifier.ann;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 11/11/15.
 */
public abstract class MyANN extends MultilayerPerceptron {

    public abstract void buildClassifier(Instances data);
    public abstract double[] distributionForInstance(Instance instance);
    public abstract double classifyInstance(Instance instance);
    public abstract Capabilities getCapabilities();
}
