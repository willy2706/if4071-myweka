import common.util.EntropyCalcUtil;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Created by nim_13512065 on 9/25/15.
 * ini adalah struktur tree
 */
public class MyJ48ClassifierTree {
    private Attribute splittedAttribute;
    private MyJ48ClassifierTree[] children;

    private Integer decisionIndex;
    private double[] classDistribution;
    private double threshold;
    private Instances _data;
    public MyJ48ClassifierTree () {
        setDecisionIndex(null);
        setChildren(null);
        setClassDistribution(null);
        setSplittedAttribute(null);
    }

    public MyJ48ClassifierTree (MyJ48ClassifierTree myJ48ClassifierTree) {
        setDecisionIndex(myJ48ClassifierTree.getDecisionIndex());
        splittedAttribute = myJ48ClassifierTree.splittedAttribute;
        int classDistributionLength = myJ48ClassifierTree.getClassDistribution().length;
        if (classDistributionLength > 0) {
            setClassDistribution(new double[classDistributionLength ]);
            for (int i = 0; i < classDistributionLength ; ++i) {
                getClassDistribution()[i] = myJ48ClassifierTree.getClassDistribution()[i];
            }

        }
        _data = myJ48ClassifierTree._data;
        int childLength = myJ48ClassifierTree.children.length;
        if (childLength > 0) {
            children = new MyJ48ClassifierTree[childLength];
            for (int i = 0; i < childLength; ++i) {
                children[i] = myJ48ClassifierTree.children[i];
            }
        }
    }

    public void buildClassifier(Instances data) throws Exception {
        this._data = data;
        if (data.numInstances() == 0) {
            setDecisionIndex(-999);
        } else {
            int numAttr = data.numAttributes();
            double[] gainRatios = new double[numAttr];
            Enumeration enumeration = data.enumerateAttributes();
            while (enumeration.hasMoreElements()) {
                Instances missingValuesReplaced = new Instances(data);
                Attribute attribute = (Attribute) enumeration.nextElement();                
                if (attribute.isNominal()) {
                    //missingValuesReplaced = replaceMissingValues(missingValuesReplaced,attribute);
                    gainRatios[attribute.index()] = EntropyCalcUtil.calcGainRatio(missingValuesReplaced, attribute);
                }
                else if(attribute.isNumeric()) {
                    setThreshold(searchThreshold(data,attribute));
                    gainRatios[attribute.index()] = EntropyCalcUtil.calcNumericGainRatio(data,attribute,threshold);
                }
            }

            int indexLargestGainRatio = Utils.maxIndex(gainRatios);
            double gainRatio;
            if(data.attribute(indexLargestGainRatio).isNominal()) {
                gainRatio=EntropyCalcUtil.calcGainRatio(data,data.attribute(indexLargestGainRatio));
            } 
            else {
                gainRatio=EntropyCalcUtil.calcNumericGainRatio(data,data.attribute(indexLargestGainRatio),threshold);
            }
            if (Utils.eq(0,gainRatio)) { //berarti ini dijadikan leaf
                setClassDistribution(new double[data.numClasses()]);
                Enumeration instanceIterator = data.enumerateInstances();
                while (instanceIterator.hasMoreElements()) {
                    Instance instance = (Instance) instanceIterator.nextElement();
                    getClassDistribution()[(int) instance.classValue()]++;
                }
                Utils.normalize(getClassDistribution());

                setDecisionIndex(Utils.maxIndex(getClassDistribution()));
            } else { //recursive part
                int numChildrenAndIndex;
                setSplittedAttribute(data.attribute(indexLargestGainRatio));
                if(data.attribute(indexLargestGainRatio).isNumeric()) {                    
                    numChildrenAndIndex = 2;
                }
                else {
                    numChildrenAndIndex = getSplittedAttribute().numValues();
                }
                setChildren(new MyJ48ClassifierTree[numChildrenAndIndex]);
                Instances[] instancesSplitted;
                if(data.attribute(indexLargestGainRatio).isNumeric()) {                    
                    instancesSplitted = EntropyCalcUtil.splitDataByNumericAttr(data, getSplittedAttribute(),threshold);
                }
                else {
                    instancesSplitted = EntropyCalcUtil.splitDataByAttr(data, getSplittedAttribute());
                }
                
                for (int i = 0; i < numChildrenAndIndex; ++i) {
                    getChildren()[i] = new MyJ48ClassifierTree();
                    getChildren()[i].buildClassifier(instancesSplitted[i]);
                }

                for (int i = 0; i < numChildrenAndIndex; ++i) {
                    if (getChildren()[i].getDecisionIndex() != null && Utils.eq(getChildren()[i].getDecisionIndex(), -999)) {
                        double[] _classDistribution = new double[_data.numClasses()];
                        Enumeration instanceIterator = _data.enumerateInstances();
                        while (instanceIterator.hasMoreElements()) {
                            Instance instance = (Instance) instanceIterator.nextElement();
                            _classDistribution[(int) instance.classValue()]++;
                        }
                        Utils.normalize(_classDistribution);
                        int _decisonIndex = Utils.maxIndex(_classDistribution);

                        getChildren()[i].setDecisionIndex(_decisonIndex);
                        getChildren()[i].setClassDistribution(_classDistribution);
                    }
                }
            }
            prune();
        }
    }

    public void prune() throws Exception {
        if (children != null) {
            double currErrorRate = this.errorRate(_data);

            double[] _classDistribution = new double[_data.numClasses()];
            Enumeration instanceIterator = _data.enumerateInstances();
            while (instanceIterator.hasMoreElements()) {
                Instance instance = (Instance) instanceIterator.nextElement();
                _classDistribution[(int) instance.classValue()]++;
            }
            Utils.normalize(_classDistribution);
            int _decisonIndex = Utils.maxIndex(_classDistribution);

            int numTrue = 0;
            int numFalse = 0;
            instanceIterator = _data.enumerateInstances();
            while (instanceIterator.hasMoreElements()) {
                Instance instance = (Instance) instanceIterator.nextElement();
                if (instance.classValue() == _decisonIndex) {
                    ++numTrue;
                } else {
                    ++numFalse;
                }
            }
            double prunedErrorRate = (double) numFalse/ (double) (numTrue + numFalse);
            if (currErrorRate > prunedErrorRate) {
                setChildren(null);
                setSplittedAttribute(null);
                setDecisionIndex(_decisonIndex);
                setClassDistribution(_classDistribution);
            }

        }
    }

    public double classifyInstance(Instance instance) {
        //jika rekursif
        if (getDecisionIndex() == null && getSplittedAttribute() != null) {
            double idxSplittedAttr;
            if(getSplittedAttribute().isNominal()) {
                idxSplittedAttr = instance.value(getSplittedAttribute());
            }
            else {
                if(instance.value(getSplittedAttribute())>=getThreshold()) {
                    idxSplittedAttr=1.0;
                }
                else {
                    idxSplittedAttr=0.0;
                }
            }
            return getChildren()[(int)idxSplittedAttr].classifyInstance(instance);
        } else return getDecisionIndex();
    }

    public double errorRate (Instances instances) {
        int numFalse = 0;
        int numTrue = 0;
        Enumeration enumeration = instances.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            Instance instance = (Instance) enumeration.nextElement();
            if (check(instance)) {
                ++numTrue;
            } else  {
                ++numFalse;
            }
        }
        return (double) numFalse/ (double) (numFalse + numTrue);
    }
    public boolean check (Instance instance) {
        double idx = instance.classValue();
        return Utils.eq(idx, this.classifyInstance(instance));
    }

    public double[] distributionForInstance(Instance instance) {
        if (getClassDistribution() == null && getSplittedAttribute() != null) {
            double idxSplittedAttr;
            if(getSplittedAttribute().isNominal()) {
                idxSplittedAttr = instance.value(getSplittedAttribute());
                if (Double.isNaN(idxSplittedAttr)) {
                    Instances[] instancesSplitted = EntropyCalcUtil.splitDataByAttr(_data, getSplittedAttribute());
                    int largestNumIdx = -1;
                    int cnt = 0;
                    for (int i = 0; i < instancesSplitted.length; ++i) {
                        int tmp = instancesSplitted[i].numInstances();
                        if (tmp > cnt) {
                            largestNumIdx = i;
                        }
                    }
                    idxSplittedAttr = largestNumIdx;
                }
            }
            else {
                double val = instance.value(getSplittedAttribute());
                if (Double.isNaN(val)) {
                    Instances[] instancesSplitted = null;
                    try {
                        instancesSplitted = EntropyCalcUtil.splitDataByNumericAttr(_data, getSplittedAttribute(), threshold);
                        int largestNumIdx = -1;
                        int cnt = 0;
                        for (int i = 0; i < instancesSplitted.length; ++i) {
                            int tmp = instancesSplitted[i].numInstances();
                            if (tmp > cnt) {
                                largestNumIdx = i;
                            }
                        }
                        idxSplittedAttr = largestNumIdx;
                    } catch (Exception e) {
                        System.out.print("ado kimaklah ini harus solve kalau ketemu masalah ini");
                        idxSplittedAttr = 0.0;
                        e.printStackTrace();
                    }
                } else {
                    if(val>=getThreshold()) {
                        idxSplittedAttr = 1.0;
                    } else {
                        idxSplittedAttr = 0.0;
                    }
                }
            }
            return getChildren()[(int)idxSplittedAttr].distributionForInstance(instance);
        } else return getClassDistribution();
    }

    public Attribute getSplittedAttribute() {
        return splittedAttribute;
    }

    public void setSplittedAttribute(Attribute splittedAttribute) {
        this.splittedAttribute = splittedAttribute;
    }

    public MyJ48ClassifierTree[] getChildren() {
        return children;
    }

    public void setChildren(MyJ48ClassifierTree[] children) {
        this.children = children;
    }

    public Integer getDecisionIndex() {
        return decisionIndex;
    }

    public void setDecisionIndex(Integer decisionIndex) {
        this.decisionIndex = decisionIndex;
    }

    public double[] getClassDistribution() {
        return classDistribution;
    }

    public void setClassDistribution(double[] classDistribution) {
        this.classDistribution = classDistribution;
    }
    
    public double getThreshold() {
        return threshold;
    }
    
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }



    private double searchThreshold(Instances data, Attribute attribute) throws Exception {
        double[] threshold = new double[data.numInstances()];
        double[] gainRatio = new double[data.numInstances()];
        for(int i=0;i<data.numInstances()-1;++i) {
            if(data.instance(i).classValue()!=data.instance(i+1).classValue()) {
                threshold[i] = (data.instance(i).value(attribute)+data.instance(i+1).value(attribute))/2;
                gainRatio[i] = EntropyCalcUtil.calcNumericGainRatio(data, attribute, threshold[i]);
                }
        }
        double result = (double) threshold[Utils.maxIndex(gainRatio)];
        return result;
    }
}
