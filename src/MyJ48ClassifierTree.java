import common.util.EntropyCalcUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

/**
 * Created by nim_13512065 on 9/25/15.
 * ini adalah struktur tree
 */
public class MyJ48ClassifierTree {
    private Attribute splittedAttribute;
    private MyJ48ClassifierTree[] children;

    private Integer decisionIndex;
    private double[] classDistribution;

    public MyJ48ClassifierTree () {
        decisionIndex = null;
        children = null;
        classDistribution = null;
    }

    public void buildClassifier(Instances data) {
        if (data.numInstances() == 0) {
            //TODO
        } else {
            int numAttr = data.numAttributes();
            if (numAttr == 1) { //berarti ini dijadikan leaf
                classDistribution = new double[data.numClasses()];
                Enumeration instanceIterator = data.enumerateInstances();
                while (instanceIterator.hasMoreElements()) {
                    Instance instance = (Instance) instanceIterator.nextElement();
                    classDistribution[(int) instance.classValue()]++;
                }
                Utils.normalize(classDistribution);

                decisionIndex = Utils.maxIndex(classDistribution);
            } else { //recursive part
                double[] gainRatios = new double[numAttr];
                Enumeration enumeration = data.enumerateAttributes();
                while (enumeration.hasMoreElements()) {
                    Attribute attribute = (Attribute) enumeration.nextElement();
                    gainRatios[attribute.index()] = EntropyCalcUtil.calcGainRatio(data, attribute);
                }

                int indexLargestGainRatio = Utils.maxIndex(gainRatios);
                splittedAttribute = data.attribute(indexLargestGainRatio);

                int numChildrenAndIndex = splittedAttribute.numValues();
                children = new MyJ48ClassifierTree[numChildrenAndIndex];
                Instances[] instancesSplitted = EntropyCalcUtil.splitDataByAttr(data, splittedAttribute);
                for (int i = 0; i < numChildrenAndIndex; ++i) {
                    children[i] = new MyJ48ClassifierTree();
                    children[i].buildClassifier(instancesSplitted[i]);
                }
            }
        }
    }


    public double classifyInstance(Instance instance) {
        //jika rekursif
        if (decisionIndex == null && splittedAttribute != null) {
            double idxSplittedAttr = instance.value(splittedAttribute);
            return children[(int)idxSplittedAttr].classifyInstance(instance);
        } else return decisionIndex;
    }

    public double[] distributionForInstance(Instance instance) {
        if (classDistribution == null && splittedAttribute != null) {
            double idxSplittedAttr = instance.value(splittedAttribute);
            return children[(int)idxSplittedAttr].distributionForInstance(instance);
        } else return classDistribution;
    }
}
