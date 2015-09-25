import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;
import weka.core.RevisionUtils;

/**
 * Created by nim_13512065 on 9/25/15.
 */
public class MyJ48ModelSelection extends ModelSelection {
    private Instances m_allData;
    public MyJ48ModelSelection (Instances allData) {
        m_allData = allData;
    }
    @Override
    public ClassifierSplitModel selectModel(Instances data) throws Exception {
        //TODO
        return null;
    }

    @Override
    public String getRevision() {
        System.out.println("ini revisi....");
        return RevisionUtils.extract("$Revision: 1.11 $");
    }


}
