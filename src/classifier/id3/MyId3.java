package classifier.id3;

import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Enumeration;

public class MyId3 extends Classifier {

    private static final long serialVersionUID = 3658199532232755963L;

    private Attribute _classAttribute;
    /**
     * Inner node's class attributes
     */
    private Attribute _splittingAttribute; // Leaf node will have null
    private MyId3[] _children;

    /**
     * Leaf node's class attributes
     */
    private double _classValue;
    private double[] _classDistribution;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // Check data, ensure MyId3 is capable to handle the data
        getCapabilities().testWithFail(data);

        // Make copy of data, don't change argument value
        Instances dataCopy = new Instances(data);
        data.deleteWithMissingClass();

        // Build tree
        buildTree(data);
    }

    private void buildTree(Instances data) {
        if (data.numInstances() == 0) {
            /**
             * Base, no instance in node
             */
            _splittingAttribute = null;
            _classValue = Instance.missingValue();
            _classDistribution = new double[data.numClasses()];
        } else {
            // Recursion part

            double[] infoGains = new double[data.numAttributes()];
            Enumeration attrIterator = data.enumerateAttributes(); // Attribute enumeration without class attribute
            while (attrIterator.hasMoreElements()) {
                Attribute att = (Attribute) attrIterator.nextElement();
                infoGains[att.index()] = calcInfoGain(data, att);
            }
            _splittingAttribute = data.attribute(Utils.maxIndex(infoGains)); // Return the first index with biggest value

            if (Utils.eq(infoGains[_splittingAttribute.index()], 0)) {
                /**
                 * Base, no info gain when split.
                 * Can be caused by all attribute used for splitting
                 */
                // TODO implement
            } else {
                /**
                 * Recursive part
                 */
                // TODO implement
            }

        }
    }

    private double calcInfoGain(Instances data, Attribute attr) {

        Instances[] splitData = splitDataByAttr(data, attr);

        double infoGain = calcEntropy(data);
        for (int i = 0; i < attr.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= (double) splitData[i].numInstances() /
                        (double) data.numInstances() * calcEntropy(splitData[i]);
            }
        }
        return infoGain;

    }

    private double calcEntropy(Instances data) {
        // Entropy is zero if no instance, prevent divided by zero
        if (data.numInstances() == 0) return 0.0;

        double[] classCounts = new double[data.numClasses()];
        Enumeration instanceIterator = data.enumerateInstances();
        int totalInstance = 0;
        while (instanceIterator.hasMoreElements()) {
            Instance inst = (Instance) instanceIterator.nextElement();
            classCounts[(int) inst.classValue()]++;
            totalInstance++;
        }
        /**
         * total = a,b,c
         * info([a,c,b]) = entropy(a,b,c)
         * info([a,b,c]) = -a/total*log(a/total) -b/total*log(b/total) - c/total*log(c/total)
         */
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            double fraction = classCounts[j] / totalInstance;
            entropy -= fraction * Utils.log2(fraction);
        }

        return entropy;
    }

    private Instances[] splitDataByAttr(Instances data, Attribute attr) {

        Instances[] splitedData = new Instances[attr.numValues()];
        for (int i = 0; i < attr.numValues(); i++) {
            splitedData[i] = new Instances(data, data.numInstances()); // initialize with data template and max capacity
        }

        Enumeration instanceIterator = data.enumerateInstances();
        while (instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            splitedData[(int) instance.value(attr)].add(instance);
        }

        for (Instances instances : splitedData) {
            instances.compactify(); // to reduce array size to fit num of instances
        }

        return splitedData;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {

        // TODO implement
        return 0.0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        // TODO implement
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    public String toString() {
        return "MyId3 is implemented based on Id3 for educational purpose";
    }
}
