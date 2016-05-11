/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi, Francois Petitjean and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdBayesOptMT Classifier
 * 
 * wdBayesOptMT.java     
 * Code written by: Nayyar Zaidi, Francois Petitjean
 * 
 * Options:
 * -------
 * 
 * -V 	Verbosity
 * -S	Structure learning (1: NB, 2:TAN, 3:KDB, 4:BN, 5:Chordalysis)
 * -P	Parameter learning (1: MAP, 2:dCCBN, 3:wCCBN, 4:eCCBN, 5: PYP)
 * -K 	Value of K for KDB.
 * 
 */
package weka.classifiers.mmall.Online.Bayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public class wdBayesOnline {

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;

	xxyDist xxyDist_;

	public BayesTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private String m_S = "NB"; // -S (1: NB, 2:TAN, 3:KDB, 4:BN, 5:Chordalysis,
	// 6:Saturated)
	private String m_P = "MAP"; // -P (1: MAP, 2:dCCBN, 3:wCCBN, 4:eCCBN)
	private int m_KDB = 1; // -K
	private long m_Chordalysis_Mem = Long.MAX_VALUE; // -F (in thousands of free
	// parameters)
	private boolean m_MVerb = false; // -V

	private double[] m_Eta0;
	RegularizationType regType = RegularizationType.None; // -R
	private double m_Lambda = 0.01; // -L

	RandomGenerator rg = null;

	private GradientsUpdater function_to_optimize;

	private BNStructure bn = null;

	private int m_Epochs = 1;

	private Instances structure;

	public ParamScheme scheme = ParamScheme.MAP;

	private double m_CenterWeights = 0.0; // -B

	private double m_InitParameters = 1.0; // -P

	private LearningListener learningListener = null;

	private static final int N_INSTANCES_OPTIM_ETA0 = 5000;
	private static final int N_INSTANCES_OPTIM_SMOOTHING = 5000;
	private static final int N_ROUNDS_ETA0 = 10;

	private static final int BUFFER_SIZE = 100000;

	public void buildClassifier(File sourceFile) throws FileNotFoundException, IOException {
		// First fill xxYDist; everybody needs it
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		initFromStructure(structure);
		learnStructure(structure, sourceFile);

		// ------------------------------------------------------------------------
		Instance row;
		this.nInstances = 0;
		while ((row = reader.readInstance(structure)) != null) {
			dParameters_.update(row);
			this.nInstances++;
		}
		System.out.println(scheme.name());

		if (scheme.name().equals("wCCBN")) {
			if (m_InitParameters == -1) {
				m_InitParameters = 1;
			}
		}

		switch (scheme) {
		case dCCBN:
		case wCCBN:
			dParameters_.setInitParameters(m_InitParameters);
			dParameters_.setCenterWeights(m_CenterWeights);
			dParameters_.allocateMemoryForParametersAndGradients();
			break;

		default:
			break;
		}

		dParameters_.computeProbabilitiesFromCounts();

		if (scheme.name().equals("dCCBN")) {
			if (m_InitParameters == -1) {
				dParameters_.setInitParametersToMAPEstimates();
			}
		}


		// ------------------------------------------------------------------------

		function_to_optimize = null;
		switch (scheme) {
		case wCCBN:
			function_to_optimize = new GradientsUpdater_w(this);
			break;
		case dCCBN:
			function_to_optimize = new GradientsUpdater_d(this);
			break;
		default:
			break;
		}

		// System.out.println(dParameters_);
		// discriminative learning
		if (function_to_optimize != null) {
			discriminativeLearning(sourceFile);
		}

		//	Instances sampleSmoothing;
		//	if (nInstances / 10 > N_INSTANCES_OPTIM_SMOOTHING) {
		//	    sampleSmoothing = sampleData(sourceFile, N_INSTANCES_OPTIM_SMOOTHING);
		//	} else {
		//	    sampleSmoothing = sampleData(sourceFile, nInstances / 10);
		//	}
		//	dParameters_.optimizeSmoothingParameter(sampleSmoothing, this);
		dParameters_.computeFinalProbabilities();
	}

	public double getM_CenterWeights() {
		return m_CenterWeights;
	}

	public void setM_CenterWeights(double m_CenterWeights) {
		this.m_CenterWeights = m_CenterWeights;
	}

	private void initFromStructure(Instances structure) {
		m_Instances = structure;
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}

		bn = new BNStructure(m_Instances, m_S, m_KDB);
		bn.setMaxNFreeParams(m_Chordalysis_Mem);
	}

	public void learnStructure(Instances structure, File sourceFile) throws IOException {

		bn.learnStructure(structure, sourceFile, rg);

		m_Order = bn.get_Order();
		m_Parents = bn.get_Parents();
		xxyDist_ = bn.get_XXYDist();
		xxyDist_.countsToProbs();

		dParameters_ = new BayesTree(nAttributes, nc, paramsPerAtt, m_Order, m_Parents, scheme);

	}

	private void discriminativeLearning(File sourceFile) throws FileNotFoundException, IOException {
		// finding step size eta0
		findEta0(sourceFile, function_to_optimize);
		//Arrays.fill(m_Eta0, 0.01);
		
		System.out.println("optimizing with eta0 = " + Arrays.toString(m_Eta0));
		int t = 0;
		double previousRMSETraining = Double.MAX_VALUE;

		for (int epoch = 0; epoch < m_Epochs; epoch++) {
			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
			Instance row;

			double logNc = Math.log(nc);
			while ((row = reader.readInstance(structure)) != null) {

				// for prequential learning
				if (learningListener != null) {
					int classValue = (int) row.classValue();
					double[] logProbs = this.distributionForInstanceNoSmoothing(row, false);
					double SE = 0.0;
					int chosenClass = 0;
					for (int i = 0; i < logProbs.length; i++) {
						if (logProbs[i] > logProbs[chosenClass]) {
							chosenClass = i;
						}
						double diff;
						if (i == classValue) {
							diff = (1.0 - Math.exp(logProbs[i]));
						} else {
							diff = (0.0 - Math.exp(logProbs[i]));
						}
						SE += diff * diff/nc;
					}
					learningListener.updated(t, -logNc - logProbs[classValue], SE, (chosenClass == classValue) ? 0.0 : 1.0);
				}

				function_to_optimize.update(row, t);
				t++;
			}

			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
			double CLL = 0.0;
			long nErrors = 0;
			int nDataPoints = 0;
			double RMSE = 0.0;
			logNc = Math.log(nc);
			while ((row = reader.readInstance(structure)) != null) {
				int classValue = (int) row.classValue();
				double[] logProbs = this.distributionForInstanceNoSmoothing(row, false);
				CLL += -logNc - logProbs[classValue];
				int chosenClass = 0;
				for (int i = 0; i < logProbs.length; i++) {
					if (logProbs[i] > logProbs[chosenClass]) {
						chosenClass = i;
					}
					double diff;
					if (i == classValue) {
						diff = (1.0 - Math.exp(logProbs[i]));
					} else {
						diff = (0.0 - Math.exp(logProbs[i]));
					}
					RMSE += diff * diff;
				}

				if (chosenClass != classValue) {
					nErrors++;
				}
				nDataPoints++;
			}
			CLL /= nDataPoints;
			double regCost = 0.0;
			switch (regType) {
			case L1:
				regCost = 0.5 * this.getLambda() * dParameters_.l1NormShiftedLeaves();
				break;
			case L2:
				regCost = 0.5 * this.getLambda() * dParameters_.l2NormShiftedLeaves();
				break;
			case None:
			}
			double cost = CLL + regCost;
			double errorRateTraining = 1.0 * nErrors / nDataPoints;
			RMSE = Math.sqrt(RMSE / nDataPoints);

			System.out.println("epoch " + epoch + ": cost=" + cost + "\ttrain error rate=" + errorRateTraining + "\ttrain RMSE=" + RMSE);
			if (Math.abs(previousRMSETraining - RMSE) < 1e-5) {
				break;
			} else {
				previousRMSETraining = RMSE;
			}
		}

	}

	private void findEta0(File sourceFile, GradientsUpdater function) {
		System.out.println("Now optimizing eta0.");
		this.m_Eta0 = new double[structure.numClasses()];
		int maxSampleSize;
		if (this.nInstances < N_INSTANCES_OPTIM_ETA0) {
			maxSampleSize = this.nInstances;
		} else {
			maxSampleSize = N_INSTANCES_OPTIM_ETA0;
		}
		try {
			Instances smallData = sampleData(sourceFile, maxSampleSize);

			Arrays.fill(m_Eta0, 0.01);
			boolean somethingChanged = true;
			for (int r = 0; somethingChanged && r < N_ROUNDS_ETA0; r++) {
				somethingChanged = false;
				for (int c = 0; c < m_Eta0.length; c++) {
					double previousEta = this.m_Eta0[c];
					final double factor = 2.0;
					double loEta = 1;
					this.m_Eta0[c] = loEta;
					double loCost = evaluateEta0(smallData, function);
					// System.out.println("cost(eta0["+c+"])="+loCost);
					double hiEta = loEta * factor;
					this.m_Eta0[c] = hiEta;
					double hiCost = evaluateEta0(smallData, function);
					// System.out.println("cost(eta0["+c+"])="+hiCost);
					if (loCost < hiCost) {
						while (loCost < hiCost) {
							hiEta = loEta;
							hiCost = loCost;
							loEta = hiEta / factor;
							this.m_Eta0[c] = loEta;
							loCost = evaluateEta0(smallData, function);
							// System.out.println("cost(eta0["+c+"])="+loCost);
						}
					} else if (hiCost < loCost) {
						while (hiCost < loCost) {
							loEta = hiEta;
							loCost = hiCost;
							hiEta = loEta * factor;
							this.m_Eta0[c] = hiEta;
							hiCost = evaluateEta0(smallData, function);
							// System.out.println("cost(eta0["+c+"])="+hiCost);
						}
					}
					this.m_Eta0[c] = loEta;
					if (this.m_Eta0[c] != previousEta) {
						somethingChanged = true;
					}
				}
				System.out.println("best eta0 so far=" + Arrays.toString(m_Eta0));

			}

		} catch (IOException e) {
			e.printStackTrace();
			System.err.println("Error while finding eta0 -> setting it to 0.01");
			Arrays.fill(m_Eta0, 0.01);
		}
	}

	private double evaluateEta0(Instances instances, GradientsUpdater function) {
		function.update(instances);// train
		double CLL = 0.0;
		double logNc = Math.log(nc);
		for (Instance inst : instances) {
			int classValue = (int) inst.classValue();
			double[] logProbs = this.distributionForInstanceNoSmoothing(inst, false);
			CLL += -logNc - logProbs[classValue];
		}
		CLL /= instances.numInstances();
		double regCost = 0.0;
		switch (regType) {
		case L1:
			regCost = 0.5 * this.getLambda() * dParameters_.l1NormShiftedLeaves();
			break;
		case L2:
			regCost = 0.5 * this.getLambda() * dParameters_.l2NormShiftedLeaves();
			break;
		case None:
		}
		double cost = CLL + regCost;
		dParameters_.resetParameters();
		return cost;
	}

	private Instances sampleData(File sourceFile, int nSamples) throws FileNotFoundException, IOException {
		int nDataPoints = this.nInstances;
		Instances res = new Instances(structure);
		RandomDataGenerator dg = new RandomDataGenerator(rg);
		int[] indexesOfSamples = dg.nextPermutation(nDataPoints, nSamples);
		Arrays.sort(indexesOfSamples);

		Instance row;
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int currentIndex = 0;
		for (int i = 0; i < indexesOfSamples.length; i++) {
			int sampleIndex = indexesOfSamples[i];
			do {
				row = reader.readInstance(structure);
				currentIndex++;
			} while (currentIndex < sampleIndex);
			res.add(row);
		}
		return res;
	}

	public double[] distributionForInstance(Instance instance, BayesNode[] nodes) {
		if (nodes == null) {
			nodes = dParameters_.findLeavesForInstance(instance);
		}
		double[] probs = null;
		switch (scheme) {
		case MAP:
			probs = logDistributionForInstance_MAP(instance, nodes);
			break;
		case wCCBN:
			probs = logDistributionForInstance_w(instance, nodes);
			break;
		case dCCBN:
			probs = logDistributionForInstance_d(instance, nodes);
			break;
		default:
			System.out.println("m_P value should be from set {MAP, dCCBN, wCCBN, eCCBN}");
			break;

		}
		SUtils.exp(probs);
		return probs;

	}

	public double[] distributionForInstance(Instance instance) {
		return distributionForInstance(instance, null);
	}

	private double[] logDistributionForInstance_MAP(Instance instance, BayesNode[] nodes) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassProbability(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			BayesNode node = nodes[u];
			int attValue = (int) instance.value(node.root.attNumber);
			for (int c = 0; c < nc; c++) {
				probs[c] += node.getFinalProbability(attValue, c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	private double[] logDistributionForInstance_w(Instance instance, BayesNode[] nodes) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassProbability(c) * dParameters_.getClassParameter(c);
		}

		// System.out.println(Arrays.toString(dParameters_.getParameters()));

		for (int u = 0; u < nAttributes; u++) {
			BayesNode node = nodes[u];

			final int attValue = (int) instance.value(node.root.attNumber);
			for (int c = 0; c < nc; c++) {
				probs[c] += node.getFinalProbability(attValue, c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	private double[] logDistributionForInstance_d(Instance instance, BayesNode[] nodes) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassParameter(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			BayesNode node = nodes[u];
			int attValue = (int) instance.value(node.root.attNumber);
			for (int c = 0; c < nc; c++) {
				probs[c] += node.getParameter(attValue, c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	private double[] distributionForInstanceNoSmoothing(Instance instance, BayesNode[] nodes, boolean exponentiate) {
		if (nodes == null) {
			nodes = dParameters_.findLeavesForInstance(instance);
		}
		double[] probs = null;
		switch (scheme) {
		case MAP:
			probs = logDistributionForInstance_MAP_no_smoothing(instance, nodes);
			break;
		case wCCBN:
			probs = logDistributionForInstance_w_no_smoothing(instance, nodes);
			break;
		case dCCBN:
			probs = logDistributionForInstance_d_no_smoothing(instance, nodes);
			break;
		default:
			System.out.println("m_P value should be from set {MAP, dCCBN, wCCBN, eCCBN}");
			break;

		}
		if (exponentiate) {
			SUtils.exp(probs);
			Utils.normalize(probs);
		}
		return probs;

	}

	private double[] distributionForInstanceNoSmoothing(Instance instance) {
		return distributionForInstanceNoSmoothing(instance, true);
	}

	private double[] distributionForInstanceNoSmoothing(Instance instance, boolean exponentiate) {
		return distributionForInstanceNoSmoothing(instance, null, exponentiate);
	}

	private double[] logDistributionForInstance_MAP_no_smoothing(Instance instance, BayesNode[] nodes) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassProbability(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			BayesNode node = nodes[u];
			int attValue = (int) instance.value(node.root.attNumber);
			for (int c = 0; c < nc; c++) {
				probs[c] += node.getProbability(attValue, c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	private double[] logDistributionForInstance_w_no_smoothing(Instance instance, BayesNode[] nodes) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassProbability(c) * dParameters_.getClassParameter(c);
		}

		// System.out.println(Arrays.toString(dParameters_.getParameters()));

		for (int u = 0; u < nAttributes; u++) {
			BayesNode node = nodes[u];
			int attValue = (int) instance.value(node.root.attNumber);

			for (int c = 0; c < nc; c++) {
				// normal case: something we've optimized for
				probs[c] += node.getParameter(attValue, c) * node.getProbability(attValue, c);
			}
		}
		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	private double[] logDistributionForInstance_d_no_smoothing(Instance instance, BayesNode[] nodes) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassParameter(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			BayesNode node = nodes[u];
			int attValue = (int) instance.value(node.root.attNumber);
			for (int c = 0; c < nc; c++) {
				probs[c] += node.getParameter(attValue, c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public RegularizationType getRegularization() {
		return regType;
	}

	public RegularizationType getRegularizationType() {
		return regType;
	}

	public double[] getEta() {
		return m_Eta0;
	}

	public double getEta0(int c) {
		return m_Eta0[c];
	}

	public double getLambda() {
		return m_Lambda;
	}

	public String getMS() {
		return m_S;
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public BayesTree getdParameters_() {
		return dParameters_;
	}

	public Instances getM_Instances() {
		return m_Instances;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public int getnAttributes() {
		return nAttributes;
	}

	public void setK(int m_K) {
		m_KDB = m_K;
	}

	public void set_m_P(String string) {
		m_P = string;
		if (m_P.length() != 0) {
			switch (m_P) {
			case "wCCBN":
				scheme = ParamScheme.wCCBN;
				break;
			case "dCCBN":
				scheme = ParamScheme.dCCBN;
				break;
			default:
				scheme = ParamScheme.MAP;
			}

		}

	}

	public void setLambda(double a) {
		m_Lambda = a;
	}

	public void set_m_S(String string) {
		m_S = string;
	}

	public void setRegularization(RegularizationType reg) {
		regType = reg;
	}

	public void setNEpochs(int m_Epochs) {
		this.m_Epochs = m_Epochs;

	}

	public void setMaxNParameters(long m_Chordalysis_Mem2) {
		this.m_Chordalysis_Mem = m_Chordalysis_Mem2;

	}

	public void setRandomGenerator(RandomGenerator rg) {
		this.rg = rg;
	}

	public double getM_InitParameters() {
		return m_InitParameters;
	}

	public void setM_InitParameters(double m_InitParameters) {
		this.m_InitParameters = m_InitParameters;
	}

	public void setLearningListener(LearningListener learningListener) {
		this.learningListener = learningListener;
	}

}
