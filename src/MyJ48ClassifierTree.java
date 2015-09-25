import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 9/25/15.
 * ini adalah struktur tree
 */
public class MyJ48ClassifierTree {
    private MyJ48ClassifierTree[] sons;

    public MyJ48ClassifierTree () {
    }

    public void buildClassifier(Instances data) {
        //TODO
    }


    public double classifyInstance(Instance instance) {
        //TODO
        return 0;
    }

    //distribusi kelas di leaf (instance) untuk kelas tertentu
    public double[] distributionForInstance(Instance instance) {
        //TODO
        return null;
    }
}
