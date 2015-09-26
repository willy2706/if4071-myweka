package classifier.id3;

import common.util.EntropyCalcUtil;
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
                infoGains[att.index()] = EntropyCalcUtil.calcInfoGain(data, att);
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
