package Penny;

import java.io.FileReader;
import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.estimators.DiscreteEstimator;

public class NB {
	public static Instances m_Instances;
	public static DiscreteEstimator m_ClassDistribution;
	public static DiscreteEstimator[][] m_Distribution;
	public int m_numInstances;
	public int m_numClasses;
	public int m_numAttributes;

	public void buildClassifier(Instances dataset) throws Exception {

		dataset = new Instances(dataset);
		dataset.deleteWithMissingClass();

		m_numInstances = dataset.numInstances();
		m_numClasses = dataset.numClasses();
		m_numAttributes = dataset.numAttributes();
		m_Instances = new Instances(dataset);

		m_Distribution = new DiscreteEstimator[m_numAttributes - 1][m_numClasses];
		m_ClassDistribution = new DiscreteEstimator(m_numClasses, true);

		Enumeration<Attribute> enumAttribute = m_Instances.enumerateAttributes();
		int attIndex = 0;
		while (enumAttribute.hasMoreElements()) {
			Attribute attribute = enumAttribute.nextElement();
			for (int i = 0; i < m_numClasses; i++) {
				m_Distribution[attIndex][i] = new DiscreteEstimator(attribute.numValues(), true);
			}
			attIndex++;
		}

		Enumeration<Instance> enumInstance = m_Instances.enumerateInstances();
		while (enumInstance.hasMoreElements()) {
			Instance instance = enumInstance.nextElement();
			upgradeClassifier(instance);
		}
	}

	public void upgradeClassifier(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		Enumeration<Attribute> enumAttribute = instance.enumerateAttributes();
		int attIndex = 0;
		while (enumAttribute.hasMoreElements()) {
			Attribute attribute = enumAttribute.nextElement();
			m_Distribution[attIndex][(int) instance.classValue()].addValue(instance.value(attribute),
					instance.weight());
			attIndex++;
		}
		m_ClassDistribution.addValue(instance.classValue(), instance.weight());

	}

	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		double probs[] = new double[m_numClasses];
		for (int i = 0; i < m_numClasses; i++) {
			probs[i] = m_ClassDistribution.getProbability(i);
			System.out.println(probs[i]);
		}

		Enumeration<Attribute> enumAtt = instance.enumerateAttributes();
		int attIndex = 0;
		double temp = 0, max = 0;
		while (enumAtt.hasMoreElements()) {
			Attribute attribute = enumAtt.nextElement();
			if (!instance.isMissing(attribute)) {
				for (int i = 0; i < m_numClasses; i++) {
					System.out.println(m_Distribution[attIndex][i]);

					temp = Math.pow(m_Distribution[attIndex][i].getProbability(instance.value(attribute)),
							instance.attribute(attIndex).weight());
					probs[i] *=  temp;
					if(probs[i] > max){
						max = probs[i];
					}
				}
			}
			attIndex ++;
		}
		
		Utils.normalize(probs);
		
		return probs;
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		FileReader frData = new FileReader("weather.nominal.arff");
		Instances data = new Instances(frData);//
		data.setClassIndex(data.numAttributes() - 1);

		NB classifier = new NB();
		classifier.buildClassifier(data);

		double[] probs = classifier.distributionForInstance(m_Instances.instance(0));
		
		System.out.println(probs[0]);
		System.out.println(probs[1]);
	}
}
