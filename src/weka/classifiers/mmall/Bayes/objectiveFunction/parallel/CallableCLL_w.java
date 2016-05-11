package weka.classifiers.mmall.Bayes.objectiveFunction.parallel;

import java.util.Arrays;
import java.util.concurrent.Callable;

import weka.classifiers.mmall.Bayes.wdBayes;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesParametersTree;
import weka.classifiers.mmall.DataStructure.xyDist;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableCLL_w implements Callable<Double>{

	private Instances instances;
	private int start;
	private int stop;
	wdBayesNode[] myNodes;
	private double[] myProbs;
	private wdBayesParametersTree dParameters;
	private int[] order;
	private int nc;
	private double[] g;
	private double mLogNC;
	private xyDist xyDist;
	private wdBayes algorithm;

	public CallableCLL_w(Instances instances, int start, int stop, int nc,wdBayesNode[] nodes, double[] myProbs, double[]g,wdBayesParametersTree dParameters, int[] order,xyDist xyDist, wdBayes algorithm) {
		this.algorithm = algorithm;
		this.instances = instances;
		this.start = start;
		this.stop = stop;
		this.nc= nc;
		this.myNodes = nodes;
		this.myProbs = myProbs;
		this.g = g;
		this.dParameters = dParameters;
		this.order = order;
		this.mLogNC = -Math.log(nc); 
		this.xyDist = xyDist;		
	}

	@Override
	public Double call() throws Exception {
		
		boolean m_Regularization = algorithm.getRegularization();
		double m_Lambda = algorithm.getLambda();
		
		double negLogLikelihood = 0.0;
		Arrays.fill(g, 0.0);
		
		for (int i = start; i <= stop; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			wdBayes.findNodesForInstance(myNodes, instance, dParameters);

			// unboxed logDistributionForInstance_w
			for (int c = 0; c < nc; c++) {
				myProbs[c] = xyDist.pp(c) * dParameters.getClassParameter(c);
			}
			
			for (int c = 0; c < nc; c++) {
				for (int u = 0; u < myNodes.length; u++) {
					wdBayesNode bNode = myNodes[u];
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c) * bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}
			
			SUtils.normalizeInLogDomain(myProbs);
			negLogLikelihood += (mLogNC - myProbs[x_C]);
			//negLogLikelihood += (- myProbs[x_C]);
			SUtils.exp(myProbs);

			// unboxed logGradientForInstance_w
			for (int c = 0; c < nc; c++) {
				if (m_Regularization) {
					negLogLikelihood += m_Lambda/2 * dParameters.getClassParameter(c) * dParameters.getClassParameter(c);
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * xyDist.pp(c)  + m_Lambda * dParameters.getClassParameter(c);
				} else {
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * xyDist.pp(c);
				}
			}
			
			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bayesNode = myNodes[u];
				for (int c = 0; c < nc; c++) {
					int index = bayesNode.getXYIndex((int) instance.value(order[u]), c);
					double probability = bayesNode.getXYCount((int) instance.value(order[u]), c);					
					double parameter = bayesNode.getXYParameter((int) instance.value(order[u]), c);
					
					if (m_Regularization) {
						negLogLikelihood += m_Lambda/2 * parameter * parameter;
						g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability + m_Lambda * parameter;
					} else {
						g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability;
					}
				}
			}			
			
		}
		return negLogLikelihood;
	}
	
}