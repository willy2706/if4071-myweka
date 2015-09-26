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
    private double _leafClassValue;
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
             * Base part, no instance in node
             */
            _splittingAttribute = null;
            // TODO consider to use parent class distribution to fill leafClassValue rather than missing value
            _leafClassValue = Instance.missingValue();
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
                 * Base part, no info gain when split.
                 * Can be caused by all attribute used for splitting
                 * Make this node become leaf node
                 */
                _splittingAttribute = null;

                // Instances class distribution in this leaf node
                _classDistribution = new double[data.numClasses()];
                Enumeration instanceIterator = data.enumerateInstances();
                while (instanceIterator.hasMoreElements()) {
                    Instance instance = (Instance) instanceIterator.nextElement();
                    _classDistribution[(int) instance.classValue()]++;
                }
                Utils.normalize(_classDistribution);
                _classAttribute = data.classAttribute();
                // The most dominant class of instances ends in this node is selected for model
                _leafClassValue = Utils.maxIndex(_classDistribution);
            } else {
                /**
                 * Recursive part
                 * Build tree
                 */
                Instances[] splitedData = EntropyCalcUtil.splitDataByAttr(data, _splittingAttribute);
                _children = new MyId3[_splittingAttribute.numValues()];
                for (int i = 0; i < _splittingAttribute.numValues(); i++) {
                    _children[i] = new MyId3();
                    _children[i].buildTree(splitedData[i]);
                }
            }

        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyId3: cannot handle missing value");
        }
        if (_splittingAttribute == null) {
            // Base, leaf node
            return _leafClassValue;
        } else {
            // Recursive until leaf node
            return _children[(int) instance.value(_splittingAttribute)].
                    classifyInstance(instance);
        }

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MyId3: cannot handle missing value");
        }

        if (_splittingAttribute == null) {
            // Base, leaf node
            return _classDistribution;
        } else {
            // Recursive until leaf node
            return _children[(int) instance.value(_splittingAttribute)].
                    distributionForInstance(instance);
        }

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
