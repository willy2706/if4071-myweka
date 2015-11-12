
package classifier.j48;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 9/25/15.
 */

public class MyJ48 extends Classifier {

    private MyJ48ClassifierTree root;

    public MyJ48() {
        root = new MyJ48ClassifierTree();
    }


    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        Instances copy = new Instances(data);
        // Missing value
        Enumeration attrIterator = data.enumerateAttributes();
        while (attrIterator.hasMoreElements()) {
            Attribute attr = (Attribute) attrIterator.nextElement();
            if(attr.isNominal()){
                AttributeStats attributeStats = copy.attributeStats(attr.index());
                int maxIndex = 0;
                for(int i=1; i<attr.numValues(); i++){
                    if(attributeStats.nominalCounts[maxIndex] < attributeStats.nominalCounts[i]){
                        maxIndex = i;
                    }
                }
                // Replace missing value with max index
                Enumeration instEnumerate = copy.enumerateInstances();
                while(instEnumerate.hasMoreElements()){
                    Instance instance = (Instance)instEnumerate.nextElement();
                    if(instance.isMissing(attr.index())){
                        instance.setValue(attr.index(),maxIndex);
                    }
                }
            } else if (attr.isNumeric()){
                AttributeStats attributeStats = copy.attributeStats(attr.index());
                double mean = attributeStats.numericStats.mean;
                if (Double.isNaN(mean)) mean = 0;
                // Replace missing value with mean
                Enumeration instEnumerate = copy.enumerateInstances();
                while(instEnumerate.hasMoreElements()){
                    Instance instance = (Instance)instEnumerate.nextElement();
                    if(instance.isMissing(attr.index())){
                        instance.setValue(attr.index(),mean);
                    }
                }
            }
        }

        root.buildClassifier(copy);
    }


    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    @Override
    public double classifyInstance(Instance instance) {
        return root.classifyInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        
        result.enable(Capabilities.Capability.MISSING_VALUES);
        
        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }
}
