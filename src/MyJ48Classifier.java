import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 9/25/15.
 */
public class MyJ48Classifier extends Classifier {

    private MyJ48ClassifierTree root;

    public MyJ48Classifier() {

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        MyJ48ModelSelection myJ48ModelSelection = new MyJ48ModelSelection(data);
        root = new MyJ48ClassifierTree(myJ48ModelSelection);
        root.buildClassifier(data);
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        //TODO
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        //TODO
        return 0;
    }
}
