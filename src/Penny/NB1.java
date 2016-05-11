package Penny;

import java.io.FileReader;
import java.util.Enumeration;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class NB1 extends AbstractClassifier{

	private static final long serialVersionUID = 1L;
	public int numInstances;
	public int numClasses;
	public int numAttributes;
	public static Instances m_Instances;

	protected double[][] xyCountMarginal;
	protected static double[] probY;
	
	public int[] yCount; // P(C)
	public int[][][] xyCount; // P(X|C)
	protected double m = 1.0;// laplace

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		m_Instances = new Instances(instances);
		numAttributes = m_Instances.numAttributes()-1;
		numClasses = m_Instances.numClasses();
		numInstances = m_Instances.numInstances();

		m_Instances.setClassIndex(numAttributes);
		m_Instances.deleteWithMissingClass();
		xyCountMarginal = new double[numClasses][numAttributes];

		// Initialize P(X|C) and P(C)
		yCount = new int[numClasses];
		xyCount = new int[numClasses][numAttributes][];
		for (int i = 0; i < numClasses; i++) {
			for (int j = 0; j < numAttributes; j++) {
				xyCount[i][j] = new int[m_Instances.attribute(j).numValues()];
			}
		}

		Enumeration<Instance> enumInst = m_Instances.enumerateInstances();
		while (enumInst.hasMoreElements()) {// every training instance
			Instance currentInst = enumInst.nextElement();
			if (!currentInst.classIsMissing()) {
				int attIndex = 0;

				Enumeration<Attribute> enumAtt = currentInst.enumerateAttributes();
				while (enumAtt.hasMoreElements()) {// every attribute
					Attribute attribute = enumAtt.nextElement();
					
					if (!currentInst.isMissing(attribute)) {
						xyCount[(int) currentInst.classValue()][attIndex][(int) currentInst.value(attribute)]++;
						xyCountMarginal[(int) currentInst.classValue()][attIndex]++;
					}
					attIndex++;
				}

				yCount[(int) currentInst.classValue()]++;
			}
		}
		printOut();
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		probY = new double[numClasses];
		for (int i = 0; i < numClasses; i++) {
			probY[i] = (double) (yCount[i] + m) / (numInstances + m * numClasses);
			System.out.println(probY[i]);
		}

		Enumeration<Attribute> enumAtt = instance.enumerateAttributes();
		int attIndex = 0;
		double[] temp = new double[numClasses];
		while (enumAtt.hasMoreElements()) {
			Attribute attribute = enumAtt.nextElement();
			if (!instance.isMissing(attribute)) {
				for (int i = 0; i < numClasses; i++) {
					temp[i] = (xyCount[i][attIndex][(int) instance.value(attIndex)] + m)
							/ (xyCountMarginal[i][attIndex] + m * numClasses);
					probY[i] *= temp[i];
				}
			}
			attIndex++;
		}
		Utils.normalize(probY);
		return probY;
	}

	public void printOut() {

		System.out.println("P(X|C): ");
		for (int j = 0; j < numAttributes; j++) {
			for (int i = 0; i < numClasses; i++) {
				System.out.print("attribute" + j + ",class" + i + ": ");
				for (int k = 0; k < xyCount[i][j].length; k++) {
					System.out.print(xyCount[i][j][k] + m + "  ");
				}
				System.out.println();
			}
			System.out.println();
		}
		System.out.println("Build Classifier Successful!\n");
	}

	public static void main(String[] args) throws Exception {

		FileReader fr = new FileReader("weather.nominal.arff");
		Instances data = new Instances(fr);
		data.setClassIndex(data.numAttributes() - 1);

		NB1 classifier = new NB1();
		classifier.buildClassifier(data);
		
		fr = new FileReader("vote.arff");
		Instances testdata = new Instances(fr);
		testdata.setClassIndex(testdata.numAttributes()-1);
		
		Evaluation eval = new Evaluation(testdata);
		eval.evaluateModel(classifier, testdata);
		
		System.out.println(eval.errorRate());
//		classifier.distributionForInstance(m_Instances.instance(2));
//		System.out.println("Predict  probability: " + probY[0] + ", " + probY[1]);
	}
}
