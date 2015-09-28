package common.util;

import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Created by Winson on 9/26/2015.
 */
public class EntropyCalcUtil {

    public static double calcInfoGain(Instances data, Attribute attr) {

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

    private static double calcInstrinsicValue (Instances data, Attribute attr) {
        Instances[] splitData = splitDataByAttr(data, attr);

        double instrinsicValue = 0.0;

        for (int i = 0; i < attr.numValues(); ++i) {
            if (splitData[i].numInstances()>0) {
                double frac = (double)splitData[i].numInstances()/(double)data.numInstances();
                instrinsicValue -= frac * Utils.log2(frac);
            }
        }

        return instrinsicValue;
    }

    public static double calcGainRatio (Instances data, Attribute attr) {
        double infogain = calcInfoGain(data, attr);
        if (Utils.eq(0.0, infogain)) return 0.0;
        return calcInfoGain(data, attr) /calcInstrinsicValue(data, attr);
    }

    public static double calcEntropy(Instances data) {
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
            if (fraction != 0) {
                entropy -= fraction * Utils.log2(fraction);
            }
        }

        return entropy;
    }

    public static Instances[] splitDataByAttr(Instances data, Attribute attr) {

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

    public static double calcNumericGainRatio(Instances data, Attribute attribute) throws Exception {
        Instances newData = new Instances(data);
        Remove remove = new Remove();
        Add add = new Add();
        remove.setAttributeIndices(Integer.toString(attribute.index()));
        remove.setInputFormat(newData);
        newData=Filter.useFilter(newData, remove);
        
        add.setAttributeIndex("last");
        add.setNominalLabels("A,B");
        add.setAttributeName("atributPengganti"+attribute.name());
        add.setInputFormat(newData);
        newData = Filter.useFilter(newData, add);
        
        double threshold;
        double gainRatio=0.0;
        for(int i=0;i<data.numInstances()-1;i++) {
            if(data.instance(i).classValue()!=data.instance(i+1).classValue()) {
                threshold = (data.instance(i).value(attribute)+data.instance(i+1).value(attribute))/2;
                newData = new Instances(replaceMissingValues(newData,attribute));
                for(int j=0;j<data.numInstances();j++) {
                    if(data.instance(j).value(attribute) >= threshold) {
                        newData.instance(j).setValue(newData.attribute("atributPengganti"+attribute.name()), "A");
                    }
                    else {
                        newData.instance(j).setValue(newData.attribute("atributPengganti"+attribute.name()), "B");
                    }
                }
                if(calcGainRatio(newData, newData.attribute("atributPengganti"+attribute.name()))>gainRatio) {
                    gainRatio = calcGainRatio(newData, attribute);
                }
            }
        }
        return gainRatio;
    }
    
    public static Instances replaceMissingValues(Instances missingValuesReplaced, Attribute attribute) {
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
