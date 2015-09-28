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
