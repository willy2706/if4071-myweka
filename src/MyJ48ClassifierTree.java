import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;

/**
 * Created by nim_13512065 on 9/25/15.
 * ini adalah struktur tree
 */
public class MyJ48ClassifierTree {
    private MyJ48ClassifierTree[] sons;
    private final ModelSelection modelSelection;

    public MyJ48ClassifierTree (ModelSelection modelSelection) {
        this.modelSelection = modelSelection;
    }

    public void buildClassifier(Instances data) {
        //TODO
    }

    /**
     * ini parameternya sesuaikan dengan kebutuhan aja
     */
    public void buildTree() {
        //TODO
    }
}
