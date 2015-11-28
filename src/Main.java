import classifier.ann.DeltaBatchRule;
import classifier.ann.DeltaRulePerceptron;
import classifier.ann.MultiLayerPerceptron;
import classifier.ann.PerceptronTrainingRule;
import classifier.id3.MyId3;
import classifier.j48.MyJ48;
import common.util.ActivationFunction;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.PrintWriter;
import java.util.Scanner;

public class Main {
    public final static String FOLDER = "dataset/";
    public static String TRAINDATASET = FOLDER + "weather.nominal.arff";
    public static String TESTDATASET = FOLDER + "test.weather.nominal.arff";
    
    public static void setTrainDataSet(String datasetName){
        TRAINDATASET = FOLDER+datasetName;
    }
    
    public static void setTestDataSet(String dataSet) {
        TESTDATASET = FOLDER+dataSet;
    }
    
    public static void main(String[] args) throws Exception {
        int input;
        String trainDataSet,testDataSet;
        Scanner reader = new Scanner(System.in);
        
        System.out.println("File train data set:");
        trainDataSet = reader.next();
        setTrainDataSet(trainDataSet);
        
        DataSource datasource = new DataSource(TRAINDATASET);
        Instances trainInstances = datasource.getDataSet();
        trainInstances.setClassIndex(trainInstances.numAttributes()-1);
        
        System.out.println("File test data set:");
        testDataSet = reader.next();
        setTestDataSet(testDataSet);
        
        datasource = new DataSource(TESTDATASET);
        Instances testInstances = datasource.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes() - 1);
        
        Evaluation evaluation = new Evaluation(trainInstances);
        //masayu.dataset1.arff
//        PerceptronTrainingRule deltaBatchRule = new PerceptronTrainingRule();
//        deltaBatchRule.setInitialWeight(new double[]{0,0,0,0});
//        deltaBatchRule.setLearningRate(0.1);
//        deltaBatchRule.setMaxIteration(10);
//        deltaBatchRule.setMomentum(0);
//        deltaBatchRule.setVerbose(true);
//        deltaBatchRule.setTerminationMseThreshold(0.01);
//        deltaBatchRule.buildClassifier(trainInstances);
//        evaluation.evaluateModel(deltaBatchRule, testInstances);

        System.out.println("Masukkan pilihan model pembelajaran:\n 1. Decision Tree Learning\n 2. Artificial Neural Network");        
        input = reader.nextInt();
        while(input<1 || input>2) {
            System.out.println("Masukan salah");
            System.out.println("Masukkan pilihan model pembelajaran:\n 1. Decision Tree Learning\n 2. Artificial Neural Network");
            input = reader.nextInt();
        }
        if(input == 1){ //DTL
            System.out.println("Masukkan algoritma pembelajaran:\n"
                    + "1. ID3\n"
                    + "2. C4.5");
            input = reader.nextInt();
            while(input<1 || input>2) {
                System.out.println("Masukkan algoritma pembelajaran:\n"
                    + "1. ID3\n"
                    + "2. C4.5");
            input = reader.nextInt();
            }
            if(input ==1) { //ID3
                System.out.println(TRAINDATASET);
                MyId3 id3 = new MyId3();
                id3.buildClassifier(trainInstances);
                evaluation.evaluateModel(id3, testInstances);
            }
            else { //C4.5
                MyJ48 j48 = new MyJ48();
                j48.buildClassifier(trainInstances);
                evaluation.evaluateModel(j48, testInstances);
            }
        } 
        
        
        else { //input == 2 , ANN
            double momentum, learningRate, terminationDeltaMSE;
            double[] initialWeight;
            int maxIteration,initialWeightMethod;
            System.out.println("Momentum :"); momentum = reader.nextDouble();
            System.out.println("LearningRate: "); learningRate = reader.nextDouble();
            System.out.println("terminationMSE: "); terminationDeltaMSE = reader.nextDouble();
            System.out.println("maxIteration: "); maxIteration = reader.nextInt();
            System.out.println("InitialWeight: 1.Random 2.Given"); initialWeightMethod = reader.nextInt();
            
            System.out.println("Masukkan pilihan algoritma pembelajaran:\n"
                   + "1. Perceptron Training Rule\n"
                   + "2. Delta Rule Batch\n"
                   + "3. Delta Rule Incremental\n"
                   + "4. Multilayer Perceptron"); 
            input = reader.nextInt();
            while(input <1 || input >4) {
                System.out.println("Masukkan pilihan algoritma pembelajaran:\n"
                   + "1. Perceptron Training Rule\n"
                   + "2. Delta Rule Batch\n"
                   + "3. Delta Rule Incremental\n"
                   + "4. Multilayer Perceptron"); 
                input = reader.nextInt();
            }
            
            switch(input){
                case 1 : {
                    System.out.println("Masukkan pilihan fungsi aktivasi:\n"
                            + "1. Sign\n"
                            + "2. Step");
                    input = reader.nextInt();
                    PerceptronTrainingRule ptr = new PerceptronTrainingRule();
                    ptr.setMomentum(momentum);
                    ptr.setLearningRate(learningRate);
                    ptr.setTerminationMseThreshold(terminationDeltaMSE);
                    ptr.setMaxIteration(maxIteration);
                    if (input == 1) {
                        ptr.setActivationFunction(ActivationFunction.SIGN);
                    } else if (input == 2) {
                        ptr.setActivationFunction(ActivationFunction.STEP);
                    }
                    if(initialWeightMethod==2) {
                        initialWeight = new double[trainInstances.numAttributes()];
                        System.out.print("Weight awal: ");
                        double weight = reader.nextDouble();
                        for(int i=0;i<trainInstances.numAttributes()+1;++i) {
                            initialWeight[i] = weight;
                        }
                    } else {
                        ptr.initWeight();
                    }
                    ptr.buildClassifier(trainInstances);
                    evaluation.evaluateModel(ptr, testInstances);
                    break;
                }
                
                case 2 : {
                    DeltaBatchRule dbr = new DeltaBatchRule();
                    if(initialWeightMethod==2) {
                        System.out.print("Weight awal: ");
                        double weight = reader.nextDouble();
                        initialWeight = new double[trainInstances.numAttributes()];
                        for(int i=0;i<trainInstances.numAttributes()+1;++i) {
                            initialWeight[i] = weight;
                        }
                        dbr.setInitialWeight(initialWeight);
                    }
                    dbr.setMomentum(momentum);
                    dbr.setLearningRate(learningRate);
                    dbr.setTerminationMseThreshold(terminationDeltaMSE);
                    dbr.setMaxIteration(maxIteration);
                    dbr.buildClassifier(trainInstances);
                    evaluation.evaluateModel(dbr, testInstances);
                    break;
                }
                
                case 3 : {
                    DeltaRulePerceptron drp = new DeltaRulePerceptron();
                    if(initialWeightMethod==2) {
                        System.out.print("Weight awal: ");
                        double weight = reader.nextDouble();
                        initialWeight = new double[trainInstances.numAttributes()];
                        for(int i=0;i<trainInstances.numAttributes()+1;++i) {
                            initialWeight[i] = weight;
                        }
                        drp.setInitialWeight(initialWeight);
                    }
                    drp.setMomentum(momentum);
                    drp.setLearningRate(learningRate);
                    drp.setTerminationMseThreshold(terminationDeltaMSE);
                    drp.setMaxIteration(maxIteration);
                    drp.buildClassifier(trainInstances);
                    evaluation.evaluateModel(drp, testInstances);
                    break;
                }
                
                case 4 : {
                    MultiLayerPerceptron mlp = new MultiLayerPerceptron();
                    if(initialWeightMethod==2) {
                        System.out.print("Weight awal: ");
                        double weight = reader.nextDouble();
                        mlp.setInitialWeight(weight);
                    }
                    mlp.setMomentum(momentum);
                    mlp.setLearningRate(learningRate);
                    mlp.setTerminationMseThreshold(terminationDeltaMSE);
                    mlp.setMaxIteration(maxIteration);
                    mlp.buildClassifier(trainInstances);
                    System.out.print("Number of hidden layer: ");
                    int numOfHiddenLayer = reader.nextInt();
                    int[] neuronPerHiddenLayer = new int[numOfHiddenLayer];
                    for (int i = 0; i < numOfHiddenLayer; i++) {
                        System.out.print("Number of neuron for hidden layer-" + (i + 1) + ": ");
                        neuronPerHiddenLayer[i] = reader.nextInt();
                    }
                    mlp.setNeuronPerHiddenLayer(neuronPerHiddenLayer);
                    evaluation.evaluateModel(mlp, testInstances);
                    break;
                }
            }
        }

        System.out.print(evaluation.toMatrixString());
        System.out.println(evaluation.toSummaryString());
        
        PrintWriter writer = new PrintWriter("result.txt", "UTF-8");
        writer.println(evaluation.toMatrixString());
        writer.println(evaluation.toSummaryString());
        writer.close();
    }
}
