package common.util;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

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

    public static double calcGainRatio(Instances data, Attribute attr) {
        // TODO implement
        return 0.0;
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
}
