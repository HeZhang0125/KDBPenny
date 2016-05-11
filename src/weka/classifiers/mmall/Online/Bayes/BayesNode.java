package weka.classifiers.mmall.Online.Bayes;

import java.util.Arrays;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;

public class BayesNode {

	protected int[][] xyCounts; // Count for x val and the y val
	protected double[][] xyProbabilities;
	protected double[][] xyFinalProbabilities;
	protected double[][] xyParameters; // Parameter indexed by x val the y val
	protected double[][] xyGradients; // Parameter indexed by x val the y val

	BayesNode[] children;

	protected int nextBranchingAttribute; // the Attribute whose values select
	// the next child
	protected int attNumber;

	protected BayesTree forest;
	protected BayesNode root;
	protected BayesNode parent;
	protected boolean subTreeInitialised = false;
	protected long totalCountSubTree = 0L;
	protected static double CONFIDENCE_PRIOR = 1;

	protected BayesNode(int attNumber, BayesTree forest) {

		this.attNumber = attNumber;
		this.root = this;
		this.parent = null;
		this.forest = forest;
		this.nextBranchingAttribute = -1;

		int nValuesForRootAttribute = forest.getNValuesForAttribute(root.attNumber);
		int nClassValues = forest.getNClassValues();
		xyCounts = new int[nValuesForRootAttribute][nClassValues];
	}

	public BayesNode(int attNumber, BayesNode parent, BayesNode root, BayesTree forest) {

		this.attNumber = attNumber;
		this.parent = parent;
		this.root = root;
		this.forest = forest;
		this.nextBranchingAttribute = -1;

		int nValuesForRootAttribute = forest.getNValuesForAttribute(root.attNumber);
		int nClassValues = forest.getNClassValues();
		xyCounts = new int[nValuesForRootAttribute][nClassValues];

	}

	public void setCount(int attValue, int classValue, int c) {
		xyCounts[attValue][classValue] = c;
	}

	public int getCount(int attValue, int classValue) {
		return xyCounts[attValue][classValue];
	}

	public void setParameter(int attValue, int classValue, double p) {
		xyParameters[attValue][classValue] = p;
	}

	public double getParameter(int attValue, int classValue) {
		return xyParameters[attValue][classValue];
	}

	public void setProbability(int attValue, int classValue, double p) {
		xyProbabilities[attValue][classValue] = p;
	}

	public double getProbability(int attValue, int classValue) {
		return xyProbabilities[attValue][classValue];
	}

	public double getProbability(Instance instance, int classValue) {
		int attValue = (int) instance.value(root.attNumber);
		return xyProbabilities[attValue][classValue];

	}

	public void setGradient(int attValue, int classValue, double g) {
		xyGradients[attValue][classValue] = g;
	}

	public void incGradientBy(int attValue, int classValue, double incG) {
		xyGradients[attValue][classValue] += incG;
	}

	public double getGradient(int attValue, int classValue) {
		return xyGradients[attValue][classValue];
	}

	public void incParameterBy(int attValue, int classValue, double incP) {
		xyParameters[attValue][classValue] += incP;
	}

	public void incCountByOne(int attValue, int classValue) {
		xyCounts[attValue][classValue]++;
		totalCountSubTree++;
	}

	public void decCountByOne(int attValue, int classValue) {
		xyCounts[attValue][classValue]--;
		totalCountSubTree--;
	}

	protected void allocateProbabilityTable() {
		xyProbabilities = new double[forest.getNValuesForAttribute(root.attNumber)][forest.getNClassValues()];
	}

	public void computeGradient_w(Instance instance, double[] probs, RegularizationType regType, double lambda, double center) {

		// if we should optimize things here
		int attValue = (int) instance.value(root.attNumber);
		int classValue = (int) instance.classValue();

		for (int c = 0; c < forest.getNClassValues(); c++) {
			double probability = getProbability(attValue, c);
			double parameter=0.0;
			double incG = (-1) * (SUtils.ind(c, classValue) - probs[c]) * probability;
			switch (regType) {
			case L1:
				parameter = getParameter(attValue, c);
				incG += lambda * probability;
				break;
			case L2:
				parameter = getParameter(attValue, c);
				incG += lambda * (parameter - center);
				break;
			case None:
				break;
			default:
				System.err.println("Regularization " + regType.name() + " not programmed");
			}
			if(Double.isNaN(incG)){
				System.err.println(SUtils.ind(c, classValue));
				System.err.println(probs[c]);
				System.err.println(probability);
				System.err.println(parameter);
			}
			incGradientBy(attValue, c, incG);
		}

	}

	public void computeGradient_d(Instance instance, double[] probs, RegularizationType regType, double lambda, double center) {

		// if we should optimize things here
		int attValue = (int) instance.value(root.attNumber);
		int classValue = (int) instance.classValue();

		for (int c = 0; c < forest.getNClassValues(); c++) {
			double parameter;
			double incG = (-1) * (SUtils.ind(c, classValue) - probs[c]);
			switch (regType) {
			case L1:
				parameter = getParameter(attValue, c);
				incG += lambda;
				break;
			case L2:
				parameter = getParameter(attValue, c);
				incG += lambda * (parameter - center);
				break;
			case None:
				break;
			default:
				System.err.println("Regularization " + regType.name() + " not programmed");
			}
			incGradientBy(attValue, c, incG);
		}

	}

	public void updateParameters(Instance instance, int t, double[] eta0, RegularizationType regularizationType, double lambda) {
		int attValue = (int) instance.value(root.attNumber);
		for (int c = 0; c < forest.getNClassValues(); c++) {
			double step = 0.0;
			switch (regularizationType) {
			case L1:
			case L2:
				step = eta0[c] / (1 + eta0[c] * lambda * t);
				break;
			case None:
			default:
				step = eta0[c] / (1 + t);
			}
			incParameterBy(attValue, c, -step * getGradient(attValue, c));
		}
		// if (parent != null) {
		// optimize the hierarchy as well
		// parent.updateParameters(instance, t, eta0, regularizationType,
		// lambda);
		// }

	}

	public void resetGradients(Instance instance) {
		if (xyGradients != null) {
			int attValue = (int) instance.value(root.attNumber);
			for (int c = 0; c < forest.getNClassValues(); c++) {
				setGradient(attValue, c, 0.0);
			}
		}
		if (parent != null) {
			parent.resetGradients(instance);
		}

	}

	public BayesNode getLeafForInstance(Instance instance) {
		if (nextBranchingAttribute == -1 || children == null) {
			// no children
			return this;
		} else {
			// children there
			int valueForBranchingAtt = (int) instance.value(nextBranchingAttribute);
			if (children[valueForBranchingAtt] == null || children[valueForBranchingAtt].totalCountSubTree == 0) {
				// this case should never happen at training time
				return this;
			} else {
				return children[valueForBranchingAtt].getLeafForInstance(instance);
			}
		}
	}

	public void updateSubTreeWithNewInstance(Instance instance, int[] parents) {
		int rootNodeValue = (int) instance.value(root.attNumber);
		int classValue = (int) instance.classValue();
		this.updateSubTreeWithNewInstance(instance, rootNodeValue, classValue, parents, 0,true);
	}

	public void unupdateSubTreeWithNewInstance(Instance instance, int[] parents) {
		int rootNodeValue = (int) instance.value(root.attNumber);
		int classValue = (int) instance.classValue();
		this.updateSubTreeWithNewInstance(instance, rootNodeValue, classValue, parents, 0,false);
	}


	private void updateSubTreeWithNewInstance(Instance instance, int rootNodeValue, int classValue, int[] parents, int indexParent,boolean increment) {

		if(increment){
			incCountByOne(rootNodeValue, classValue);
		}else{
			decCountByOne(rootNodeValue, classValue);
		}

		if (parents != null && indexParent < parents.length) {

			if (nextBranchingAttribute == -1) {
				nextBranchingAttribute = parents[indexParent];
				children = new BayesNode[forest.getNValuesForAttribute(nextBranchingAttribute)];
			}
			int existingChildValue = (int) instance.value(nextBranchingAttribute);
			for (int v = 0; v < children.length; v++) {
				if (children[v] == null) {
					children[v] = new BayesNode(nextBranchingAttribute, this, this.root, this.forest);
				}
				if (v == existingChildValue) {
					children[v].updateSubTreeWithNewInstance(instance, rootNodeValue, classValue, parents, indexParent + 1,increment);
				} else {
					children[v].updateSubTreeWithNoInstance(parents, indexParent + 1);
				}
			}
		}
		subTreeInitialised = true;
	}

	private void updateSubTreeWithNoInstance(int[] parents, int indexParent) {
		if (!subTreeInitialised && parents != null && indexParent < parents.length) {
			if (nextBranchingAttribute == -1) {
				nextBranchingAttribute = parents[indexParent];
				children = new BayesNode[forest.getNValuesForAttribute(nextBranchingAttribute)];
			}
			for (int v = 0; v < children.length; v++) {
				if (children[v] == null) {
					children[v] = new BayesNode(nextBranchingAttribute, this, this.root, this.forest);
				}
				children[v].updateSubTreeWithNoInstance(parents, indexParent + 1);
			}
		}
		subTreeInitialised = true;
	}

	public boolean isLeaf() {
		return children == null;
	}

	public boolean hasSeenAllChildrenAtLeastOnce() {
		for (int i = 0; i < children.length; i++) {
			if (children[i] == null) {
				return false;
			}
		}
		return true;
	}

	public void allocateMemoryForParametersAndGradients() {
		if (children == null) {
			int nc = forest.getNClassValues();
			int nval = forest.getNValuesForAttribute(root.attNumber);
			xyParameters = new double[nval][nc];
			resetParameters();
			xyGradients = new double[nval][nc];
		} else {
			for (int i = 0; i < children.length; i++) {
				if (children[i] != null) {
					children[i].allocateMemoryForParametersAndGradients();
				}
			}
		}
	}

	public void computeProbabilitiesFromCounts() {
		int nc = forest.getNClassValues();
		int nval = forest.getNValuesForAttribute(root.attNumber);
		allocateProbabilityTable();
		for (int c = 0; c < nc; c++) {
			long denom = 0;
			for (int v = 0; v < nval; v++) {
				denom += getCount(v, c);
			}
			for (int v = 0; v < nval; v++) {
				double prob = Math.log(Math.max(SUtils.MEsti(getCount(v, c), denom, nval), 1e-75));
				setProbability(v, c, prob);
			}
		}
		for (int i = 0; children != null && i < children.length; i++) {
			if (children[i] != null) {
				children[i].computeProbabilitiesFromCounts();
			}
		}
	}
	
	public void setInitParametersToMAPEstimates() {
		int nc = forest.getNClassValues();
		int nval = forest.getNValuesForAttribute(root.attNumber);
		
		for (int c = 0; c < nc; c++) {
			for (int v = 0; v < nval; v++) {
				xyParameters[v][c] = getProbability(v, c);
			}
		}
		
		for (int i = 0; children != null && i < children.length; i++) {
			if (children[i] != null) {
				children[i].setInitParametersToMAPEstimates();
			}
		}
	}
	
	public void resetParameters() {
		if (xyParameters != null) {
			for (int i = 0; i < xyParameters.length; i++) {
				Arrays.fill(xyParameters[i], forest.getInitParameters());
			}
		}
		for (int i = 0; children != null && i < children.length; i++) {
			if (children[i] != null) {
				children[i].resetParameters();
			}
		}
	}

	public String toString(String prefix) {
		String str = prefix + attNumber + " branching on " + nextBranchingAttribute + "\t" + xyGradients + "\n";
		if (children != null) {
			for (int i = 0; i < children.length; i++) {
				BayesNode n = children[i];
				if (n != null)
					str += n.toString(prefix + "\t");

			}
			str += "\n";
		}
		return str;
	}

	/**
	 * Computes the squared L2 norm of the parameter vector, shifted by the
	 * center parameter (||w-mu||_2)^2
	 * 
	 * @return
	 */
	public double squaredl2normShiftedLeaves() {
		if (hasParameter()) {
			double res = 0.0;
			for (int v = 0; v < xyParameters.length; v++) {
				long nTimesSeen = 0L;
				for (int c = 0; nTimesSeen == 0 && c < xyParameters[v].length; c++) {
					nTimesSeen += getCount(v, c);
				}
				if (nTimesSeen > 0) {
					for (int c = 0; c < xyParameters[v].length; c++) {
						double diff = getParameter(v, c) - forest.getCenterWeights();
						res += diff * diff;
					}
				}
			}
			return res;
		} else {
			double res = 0.0;
			for (int i = 0; i < children.length; i++) {
				if (children[i] != null) {
					res += children[i].squaredl2normShiftedLeaves();
				}
			}

			return res;
		}
	}

	public double l1normShiftedLeaves() {
		if (hasParameter()) {
			double res = 0.0;
			for (int v = 0; v < xyParameters.length; v++) {
				long nTimesSeen = 0L;
				for (int c = 0; nTimesSeen == 0 && c < xyParameters[v].length; c++) {
					nTimesSeen += getCount(v, c);
				}
				if (nTimesSeen > 0) {
					for (int c = 0; c < xyParameters[v].length; c++) {
						res += Math.abs(getParameter(v, c) - forest.getCenterWeights());
					}
				}
			}
			return res;
		} else {
			double res = 0.0;
			for (int i = 0; i < children.length; i++) {
				if (children[i] != null) {
					res += children[i].l1normShiftedLeaves();
				}
			}

			return res;
		}

	}

	protected boolean hasParameter() {
		return xyParameters != null;
	}

	public double meanParameterChildren(int attValue, int c) {
		double res = 0.0;
		for (int i = 0; i < children.length; i++) {
			if (children[i] != null) {
				long tmpW = children[i].getCount(attValue, c);
				if (children[i].hasParameter()) {
					res += tmpW * children[i].getParameter(attValue, c);
				} else {
					// marginalizing deeper
					res += tmpW * children[i].meanParameterChildren(attValue, c);
				}
			}
		}
		return res / getCount(attValue, c);
	}

	public void computeFinalProbabilities() {
		if (totalCountSubTree == 0)
			return;
		if (parent == null) {
			xyFinalProbabilities = new double[xyProbabilities.length][xyProbabilities[0].length];
			// root node - p(x_k|y) - smoothing with p(x_k)
			for (int v = 0; v < xyProbabilities.length; v++) {
				double probaPrior = 0.0;// p(x_k)
				for (int c = 0; c < xyProbabilities[v].length; c++) {
					probaPrior += Math.exp(xyProbabilities[v][c]);
				}
				for (int c = 0; c < xyProbabilities[v].length; c++) {
					long countY = forest.getClassCount(c);// #(y)
					double probaCurrent = Math.exp(xyProbabilities[v][c]);// p(x_k|y)
					xyFinalProbabilities[v][c] = Math.log((countY * probaCurrent) / (countY + CONFIDENCE_PRIOR) + (CONFIDENCE_PRIOR * probaPrior)
							/ (countY + CONFIDENCE_PRIOR));
				}
			}
		} else {
			xyFinalProbabilities = new double[xyProbabilities.length][];
			// not a root node - p(x_k|y,x1,x2) - smoothing with parent, ie
			// p(x_k|y,x1)
			for (int v = 0; v < xyProbabilities.length; v++) {

				int countForXk = getCount(v, 0);//#(x_k,x1,x2)
				int previousCount = countForXk;
				boolean allValuesForClassIdentical = true; // will store if we're able to discriminate between classes
				for (int c = 1; allValuesForClassIdentical && c < xyProbabilities[v].length; c++) {
					int count = getCount(v, c);
					allValuesForClassIdentical = (previousCount==count);
					countForXk += count;
				}
				if (allValuesForClassIdentical) {
					/* if we have seen the same number of times all the classes, 
					 * then we backoff top(x_k|y,x1) because it means we can't 
					 * discriminate between them  
					 */
					xyFinalProbabilities[v] = parent.xyFinalProbabilities[v];
				} else {
					xyFinalProbabilities[v] = new double[xyProbabilities[v].length];
					for (int c = 0; c < xyProbabilities[v].length; c++) {
						// int countForConditioning = 0;
						// for (int v1 = 0; v1 < xyProbabilities.length; v1++) {
						// countForConditioning += getCount(v1, c);
						// }
						// #(y,x1,x2) - marginalising over x_k
						double probaPrior = Math.exp(parent.xyFinalProbabilities[v][c]);// p(x_k|y,x1)
						double probaCurrent = Double.NaN;// p(x_k|y,x1,x2)
						if (!hasParameter()) {// not optimised => MAP
							probaCurrent = Math.exp(xyProbabilities[v][c]);
						} else {
							switch (forest.getScheme()) {
							case MAP:// shouldn't happen as no parameter
								// array
								probaCurrent = Math.exp(xyProbabilities[v][c]);
								break;
							case dCCBN:
								probaCurrent = Math.exp(xyParameters[v][c]);
								break;
							case wCCBN:
								probaCurrent = Math.exp(xyParameters[v][c] * xyProbabilities[v][c]);
								break;
							}
						}
						double thisConfidence = countForXk;
						// double thisConfidence = countForConditioning;
						xyFinalProbabilities[v][c] = Math.log((thisConfidence * probaCurrent) / (thisConfidence + CONFIDENCE_PRIOR)
								+ (CONFIDENCE_PRIOR * probaPrior) / (thisConfidence + CONFIDENCE_PRIOR));
						// if(Math.abs(xyFinalProbabilities[v][c]-xyProbabilities[v][c])>0.01){
						// System.out.println(Math.exp(xyProbabilities[v][c])+" => "+Math.exp(xyFinalProbabilities[v][c]));
						// }
					}
				}
			}
		}
		if (children != null) {
			for (int c = 0; c < children.length; c++) {
				children[c].computeFinalProbabilities();
			}
		}

	}

	public double getFinalProbability(int attValue, int c) {
		return xyFinalProbabilities[attValue][c];
	}
}