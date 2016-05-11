package weka.classifiers.mmall.Evaluation;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Random;

import org.apache.commons.math3.random.MersenneTwister;

import weka.classifiers.mmall.Online.Bayes.LearningListener;
import weka.classifiers.mmall.Online.Bayes.RegularizationType;
import weka.classifiers.mmall.Online.Bayes.wdBayesOnline;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;

public class PrequentialOOCWdBayes {

	private static String data = "";
	private static String m_OutputResults = "";

	private static String m_S = "NB"; // -S (NB, TAN, KDB, Chordalysis)
	private static String m_P = "MAP"; // -P (MAP, dCCBN, wCCBN, eCCBN)

	private static int m_K = 1; // -K

	private static boolean m_MVerb = false; // -V

	private static RegularizationType m_Regularization = RegularizationType.None; // -R
	private static int m_Epochs = 1; // -A
	private static double m_Lambda = 0.01; // -L

	private static long m_Chordalysis_Mem = Long.MAX_VALUE; // -F (in thousands
	// of free
	// parameters)

	private static double m_CenterWeights; 
	private static double m_InitParameters = 1.0; // -P (initialize weights, MAP is set to -1)

	private static int m_nExp = 100; // -X (no. of experiments)

	private static int m_StepSizePlot = 1000; // -Q (buffer size)

	private static int m_Listner = 1; // -Z (listner, 1 or 2)

	public static void main(String[] args) throws Exception {

		setOptions(args);

		if (data.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);
		}

		File sourceFile;
		sourceFile = new File(data);
		if (!sourceFile.exists()) {
			System.err.println("File " + data + " not found!");
			System.exit(-1);
		}

		long seed = 3071980;

		System.out.println("Interval Step Size is: " + m_StepSizePlot);

		/*
		 * Read file sequentially, 10000 instances at a time
		 */
		ArffLoader loader = new ArffLoader();
		loader.setFile(sourceFile);
		Instances dataset = loader.getDataSet();

		for (int s = 0; s < 20; s++) {
			System.out.println("pre shuffling iteration " + s);
			dataset.randomize(new Random(seed + s));
		}

		double[] avgErrorRates = null;
		double[] avgRMSEs = null;
		LearningListener listener = null;

		if (m_Listner == 1) {
			avgErrorRates = new double[m_Epochs*dataset.size() / m_StepSizePlot];
			avgRMSEs = new double[m_Epochs*dataset.size() / m_StepSizePlot];

		} else if (m_Listner == 2) {
			avgErrorRates = new double[(m_Epochs*dataset.size()) - (2 * m_StepSizePlot)];
			avgRMSEs = new double[(m_Epochs*dataset.size()) - (2 * m_StepSizePlot)];
		}

		for (int exp = 0; exp < m_nExp; exp++) {
			System.out.println("shuffling for exp " + exp);

			dataset.randomize(new Random(seed + exp));

			File tmpDataFile = File.createTempFile(sourceFile.getName(), ".arff");
			ArffSaver fileSaver = new ArffSaver();
			fileSaver.setFile(tmpDataFile);
			fileSaver.setDestination(tmpDataFile);
			fileSaver.setRetrieval(ArffSaver.BATCH);
			fileSaver.setInstances(dataset);
			fileSaver.writeBatch();

			// Tracks how many times each class was predicted for each instance

			MersenneTwister rg = new MersenneTwister(seed + exp);

			wdBayesOnline learner = new wdBayesOnline();
			learner.set_m_S(m_S);
			learner.set_m_P(m_P);
			learner.setRegularization(m_Regularization);
			learner.setLambda(m_Lambda);
			learner.setK(m_K);
			learner.setNEpochs(m_Epochs);
			learner.setMaxNParameters(m_Chordalysis_Mem);
			learner.setRandomGenerator(rg);
			learner.setM_CenterWeights(m_CenterWeights);
			learner.setM_InitParameters(m_InitParameters);

			if (m_Listner == 1) {

				listener = new LearningListenerCollector(m_StepSizePlot);

			} else if (m_Listner == 2) {

				listener = new LearningListenerCollector2(m_StepSizePlot);
			}

			learner.setLearningListener(listener);

			learner.buildClassifier(tmpDataFile);

			tmpDataFile.delete();

			ArrayList<Double> errorRates = listener.getErrorRates();
			ArrayList<Double> rmses = listener.getRMSEs();

			System.out.println("Size of errorRates is: " +  errorRates.size() + " and, avgErroRates is: " + ((m_Epochs*dataset.size()) - (2 * m_StepSizePlot)));
			for (int i = 0; i < errorRates.size(); i++) {
				avgErrorRates[i] += errorRates.get(i);
				avgRMSEs[i] += rmses.get(i);
			}
		}

		// Outputting stats
		File logFile;
		if (m_OutputResults.isEmpty()) {
			logFile = File.createTempFile("log-", ".csv");
		} else {
			logFile = new File(m_OutputResults);
		}
		System.out.println("Logging to " + logFile.getAbsolutePath());

		// Outputting stats
		PrintWriter out = new PrintWriter(new BufferedOutputStream(new FileOutputStream(logFile)));
		//out.println("nInstances,avg error rate, avg rmse");
		for (int i = 0; i < avgErrorRates.length; i++) {
			avgErrorRates[i] /= m_nExp;
			avgRMSEs[i] /= m_nExp;
			out.println((i+1)*m_StepSizePlot+","+avgErrorRates[i]+","+avgRMSEs[i]);
		}
		out.flush();
		out.close();

	}

	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static void setOptions(String[] options) throws Exception {

		String Strain = Utils.getOption('t', options);
		if (Strain.length() != 0) {
			data = Strain;
		}

		String Soutput = Utils.getOption('O', options);
		if (Soutput.length() != 0) {
			m_OutputResults = Soutput;
		}

		m_MVerb = Utils.getFlag('V', options);

		String Mem = Utils.getOption('F', options);
		if (Mem.length() != 0) {
			m_Chordalysis_Mem = Long.parseLong(Mem);
		}

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			m_S = SK;
		}

		String MP = Utils.getOption('M', options);
		if (MP.length() != 0) {
			m_P = MP;
		}

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_K = Integer.parseInt(MK);
		}

		String MR = Utils.getOption('R', options);
		if (MR.length() != 0) {
			switch (MR) {
			case "L1":
				m_Regularization = RegularizationType.L1;
				break;
			case "L2":
				m_Regularization = RegularizationType.L2;
				break;
			default:
				m_Regularization = RegularizationType.None;
			}
		} else {
			m_Regularization = RegularizationType.None;
		}

		String strA = Utils.getOption('A', options);
		if (strA.length() != 0) {
			m_Epochs = (Integer.valueOf(strA));
		}

		String strL = Utils.getOption('L', options);
		if (strL.length() != 0) {
			m_Lambda = (Double.valueOf(strL));
		}

		String strB = Utils.getOption('B', options);
		if (strB.length() != 0) {
			m_CenterWeights = (Double.valueOf(strB));
		}

		String strP = Utils.getOption('P', options);
		if (strP.length() != 0) {
			m_InitParameters = (Double.valueOf(strP));
		}

		String strX = Utils.getOption('X', options);
		if (strX.length() != 0) {
			m_nExp = Integer.valueOf(strX);
		}

		String strQ = Utils.getOption('Q', options);
		if (strQ.length() != 0) {
			m_StepSizePlot = Integer.valueOf(strQ);
		}

		String strZ = Utils.getOption('Z', options);
		if (strZ.length() != 0) {
			m_Listner = Integer.valueOf(strZ);
		}

		Utils.checkForRemainingOptions(options);

	}

	public static String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

}
