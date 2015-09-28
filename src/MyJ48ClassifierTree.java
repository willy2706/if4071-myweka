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
                missingValuesReplaced = replaceMissingValues(missingValuesReplaced,attribute);
                if (attribute.isNominal()) {
                    gainRatios[attribute.index()] = EntropyCalcUtil.calcGainRatio(missingValuesReplaced, attribute);
                }
                else if(attribute.isNumeric()) {
                    gainRatios[attribute.index()] = EntropyCalcUtil.calcNumericGainRatio(data,attribute);
                }
            }

            int indexLargestGainRatio = Utils.maxIndex(gainRatios);
            if (Utils.eq(0,EntropyCalcUtil.calcGainRatio(data,data.attribute(indexLargestGainRatio)))) { //berarti ini dijadikan leaf
                setClassDistribution(new double[data.numClasses()]);
                Enumeration instanceIterator = data.enumerateInstances();
                while (instanceIterator.hasMoreElements()) {
                    Instance instance = (Instance) instanceIterator.nextElement();
                    getClassDistribution()[(int) instance.classValue()]++;
                }
                Utils.normalize(getClassDistribution());

                setDecisionIndex(Utils.maxIndex(getClassDistribution()));
            } else { //recursive part
                setSplittedAttribute(data.attribute(indexLargestGainRatio));

                int numChildrenAndIndex = getSplittedAttribute().numValues();
                setChildren(new MyJ48ClassifierTree[numChildrenAndIndex]);
                Instances[] instancesSplitted = EntropyCalcUtil.splitDataByAttr(data, getSplittedAttribute());
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
            double idxSplittedAttr = instance.value(getSplittedAttribute());
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
            double idxSplittedAttr = instance.value(getSplittedAttribute());
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

    public Instances replaceMissingValues(Instances missingValuesReplaced, Attribute attribute) {
        double missingValueClass=0.0;
        int[] classes = new int[missingValuesReplaced.numDistinctValues(attribute)];
        int[] max = new int[missingValuesReplaced.numClasses()];
        for(int i=0;i<classes.length;i++) {
            classes[i]=0;
        }
        Instances newInstances = new Instances(missingValuesReplaced);
        for(int i=0;i<missingValuesReplaced.numInstances();i++) {
            if(missingValuesReplaced.instance(i).isMissing(attribute)) {
                missingValueClass = missingValuesReplaced.instance(i).classValue();
                for(int j=0;j<missingValuesReplaced.numInstances();++j) {
                    if(!missingValuesReplaced.instance(j).isMissing(attribute) && 
                            missingValuesReplaced.instance(j).classValue()==missingValueClass) {
                        classes[(int)missingValuesReplaced.instance(j).value(attribute)]++;
                    }
                }
                //cari max dari tabel classes
                int maxAttributes=0;
                for(int j=0;j<classes.length;j++) {
                    if(classes[j]>=maxAttributes) {
                        maxAttributes = j;
                    }
                }
                //untuk yang kelasnya yes, maxnya adalah atribut maxAttributes
                max[(int)missingValuesReplaced.instance(i).classValue()] = maxAttributes;
            }
            //kosongin tabelnya lagi
            for(int j=0;j<classes.length;j++) {
                classes[j]=0;
            }
        }
        for(int i=0;i<missingValuesReplaced.numInstances();i++) {
            if(newInstances.instance(i).isMissing(attribute)) {
                newInstances.instance(i).setValue(attribute, max[(int)newInstances.instance(i).classValue()]);
            }
        }
        return newInstances;
    }
}
