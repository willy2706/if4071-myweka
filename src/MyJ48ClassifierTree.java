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

    public void buildClassifier(Instances data) throws Exception {
        if (data.numInstances() == 0) {
            System.out.println("todo woi");
            //TODO ini kalo di root, jadi ga bisa di build classifiernya
            // tapi kalo di leaf bakal di hitung pake probabilitas yang rumusnya belum ketemu
        } else {
            int numAttr = data.numAttributes();
            double[] gainRatios = new double[numAttr];
            Enumeration enumeration = data.enumerateAttributes();
            while (enumeration.hasMoreElements()) {
                Attribute attribute = (Attribute) enumeration.nextElement();
                if (attribute.isNominal()) {
                    gainRatios[attribute.index()] = EntropyCalcUtil.calcGainRatio(data, attribute);
                }
                else if(attribute.isNumeric()) {
                    gainRatios[attribute.index()] = EntropyCalcUtil.calcNumericGainRatio(data,attribute);
                }
            }

            int indexLargestGainRatio = Utils.maxIndex(gainRatios);
            if (Utils.eq(0,EntropyCalcUtil.calcGainRatio(data,data.attribute(indexLargestGainRatio)))) { //berarti ini dijadikan leaf
                classDistribution = new double[data.numClasses()];
                Enumeration instanceIterator = data.enumerateInstances();
                while (instanceIterator.hasMoreElements()) {
                    Instance instance = (Instance) instanceIterator.nextElement();
                    classDistribution[(int) instance.classValue()]++;
                }
                Utils.normalize(classDistribution);

                decisionIndex = Utils.maxIndex(classDistribution);
            } else { //recursive part
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
